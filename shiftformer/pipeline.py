"""Transformation x normalization selection with Transformer denoise (v1).

    result = shiftformer.choose_combination(train, test)
    result.summary      # DataFrame, 30 rows
    result.best_by_pearson, result.best_by_spearman, result.final, result.report, result.log
    result.save(out_dir)   # summary.tsv, report.txt, run_log.json

train : raw count matrix (no dropout / normalization / transformation)
test  : externally generated dropout matrix of the same cells and genes
Each may be a pandas DataFrame, an AnnData, or a file path (.feather/.parquet/.csv/.tsv).
The package reads no other paths and never applies dropout itself.

To spread the 30 combinations over several jobs:
    part = choose_combination(train, test, combinations=[i])   # i in 0..29
    part.save(dir_i); ...; merge_results([load_result(d) for d in dirs]).save(out_dir)
"""
import datetime as _dt
import hashlib
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split

from . import cv, metrics
from . import preprocess as P

# Fixed hyperparameters (= Transformer_denoise settings of the original analysis)
DEFAULT_HP = {
    "n_tokens": 32, "d_model": 64, "nhead": 4, "num_layers": 1, "dim_feedforward": 128,
    "dropout": 0.1, "lr": 1e-4, "batch_size": 64, "max_epochs": 100, "patience": 10,
    "min_delta": 0.001, "check_every": 1, "zero_weight": 1.0, "nonzero_weight": 1.0,
}
CONSTANT_RULE = ("a gene's correlation is undefined if, across cells, the reference vector or the "
                 "prediction vector is constant (max == min exactly, evaluated in float64 on the "
                 "reconstruction cast to the saving precision) or there are < 2 cells; undefined -> 0; "
                 "mean over all genes")
SUMMARY_COLUMNS = ["transformation", "normalization", "mean_pearson", "mean_spearman",
                   "n_undefined_pearson", "n_undefined_spearman"]


class StopForUser(RuntimeError):
    """Data problem the user must decide on. Nothing was dropped or replaced.
    `.log` holds the run log (with stop_reason / stop_details)."""

    def __init__(self, message, log=None):
        super().__init__(message)
        self.log = log or {}


def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _describe_input(x):
    return str(x) if isinstance(x, (str, os.PathLike)) else f"<{type(x).__name__}>"


# =================================================================== result object
class ChooseCombinationResult:
    """summary, best_by_pearson, best_by_spearman, final, report, log, reconstructions (optional)."""

    def __init__(self, log, records, reconstructions=None):
        self.log = log
        self.records = records                    # {combo index: per-combination record}
        self.reconstructions = reconstructions or {}
        self._finalize()

    def _finalize(self):
        rows, per_combo = [], []
        for i, (t, n) in enumerate(P.all_combos()):
            r = self.records.get(i, {"status": "not_run"})
            ok = r["status"] == "success"
            rows.append({"transformation": t, "normalization": n,
                         "mean_pearson": r["mean_pearson"] if ok else np.nan,
                         "mean_spearman": r["mean_spearman"] if ok else np.nan,
                         "n_undefined_pearson": r["n_undefined_pearson"] if ok else pd.NA,
                         "n_undefined_spearman": r["n_undefined_spearman"] if ok else pd.NA,
                         "status": r["status"]})
            per_combo.append({"index": i, "transformation": t, "normalization": n, **r})
        df = pd.DataFrame(rows)
        for c in ("n_undefined_pearson", "n_undefined_spearman"):
            df[c] = df[c].astype("Int64")
        self.n_successful = int((df.status == "success").sum())
        self.final = self.n_successful == 30
        self.summary = df[SUMMARY_COLUMNS].copy()
        self.status = df[["transformation", "normalization", "status"]].copy()

        def best(col):
            ok = df[df.status == "success"]
            if ok.empty:
                return None
            b = float(ok[col].max())
            tied = ok[ok[col] == b]                      # exact float64 equality
            return {"score": b, "final": self.final,
                    "combinations": [{"transformation": t, "normalization": n}
                                     for t, n in zip(tied.transformation, tied.normalization)]}
        self.best_by_pearson, self.best_by_spearman = best("mean_pearson"), best("mean_spearman")

        label = "FINAL best" if self.final else "PROVISIONAL leader (successful combinations only)"
        d, gs = self.log["data"], self.log["data"]["library_size_gene_set"]
        lines = ["Transformation-normalization selection (Transformer denoise, v1)",
                 f"cells={d['n_cells_used']}  genes={d['n_genes_used']}  "
                 f"all-zero reference genes={d['n_all_zero_reference_genes']}  "
                 f"seed={self.log['split']['seed']}  folds={self.log['split']['n_folds']}",
                 f"library-size gene set vs original pipeline: {gs['vs_original_pipeline']} - {gs['statement']}",
                 f"combinations successful: {self.n_successful}/30",
                 "scores: float64, printed with full precision (repr); ties = exactly equal float64 scores"]
        if not self.final:
            lines.append("FAILED / NOT RUN (NA in summary):")
            for c in per_combo:
                if c["status"] != "success":
                    lines.append(f"  {c['transformation']} | {c['normalization']}: {c['status']}"
                                 + (f" - {c['error']}" if c.get("error") else ""))
            lines.append(f"PROVISIONAL ONLY: leaders among the {self.n_successful} successful combinations. "
                         "This is NOT the final best over all 30 combinations.")
        for name, b in (("Pearson", self.best_by_pearson), ("Spearman", self.best_by_spearman)):
            if b is None:
                lines.append(f"{label} by mean gene-wise {name}: none (no successful combination)")
                continue
            k = len(b["combinations"])
            lines.append(f"{label} by mean gene-wise {name}: {b['score']!r}"
                         + (f"  [{k} combinations exactly tied]" if k > 1 else ""))
            lines += [f"  transformation={c['transformation']}  normalization={c['normalization']}"
                      for c in b["combinations"]]
        self.report = "\n".join(lines) + "\n"

        self.log.update(
            status="complete" if self.final else "incomplete", finished=_now(),
            n_combinations_successful=self.n_successful,
            selection={"final": self.final, "label": label,
                       "by_pearson": self.best_by_pearson, "by_spearman": self.best_by_spearman},
            combinations=[{k: v for k, v in c.items() if k != "traceback"} for c in per_combo])

    def save(self, out_dir):
        """Write summary.tsv, report.txt, run_log.json (reconstructions are not written)."""
        os.makedirs(out_dir, exist_ok=True)
        self.summary.to_csv(os.path.join(out_dir, "summary.tsv"), sep="\t", index=False,
                            float_format="%.17g", na_rep="NA")
        with open(os.path.join(out_dir, "report.txt"), "w") as f:
            f.write(self.report)
        with open(os.path.join(out_dir, "run_log.json"), "w") as f:
            json.dump(self.log, f, indent=2, default=str)
        return out_dir

    def __repr__(self):
        return self.report


def load_result(out_dir):
    """Rebuild a result from a directory written by .save() (e.g. to merge array parts)."""
    with open(os.path.join(out_dir, "run_log.json")) as f:
        log = json.load(f)
    records = {c["index"]: {k: v for k, v in c.items()
                            if k not in ("index", "transformation", "normalization")}
               for c in log["combinations"] if c["status"] != "not_run"}
    return ChooseCombinationResult(log, records)


def merge_results(results):
    """Combine results computed on the SAME data/split/settings for different combinations."""
    if not results:
        raise ValueError("no results to merge")
    fp = results[0].log["fingerprint"]
    records = {}
    for r in results:
        if r.log["fingerprint"] != fp:
            raise ValueError("results come from different data, splits or settings; cannot merge")
        for i, rec in r.records.items():
            if i in records and records[i].get("status") == "success" and rec.get("status") == "success":
                raise ValueError(f"combination {i} present in more than one result")
            if i not in records or rec.get("status") == "success":
                records[i] = rec
    log = json.loads(json.dumps(results[0].log, default=str))
    log["merged_from_n_results"] = len(results)
    return ChooseCombinationResult(log, records)


# =================================================================== main function
def choose_combination(train, test, orientation="cells_x_genes", combinations=None, seed=42,
                       n_folds=5, inner_val_frac=0.1, save_dtype="float32", pipeline_genes=None,
                       device="auto", threads=None, hp=None, return_reconstructions=False,
                       verbose=True):
    """Run the v1 selection and return a ChooseCombinationResult.

    train / test   : raw and dropout count matrices (DataFrame, AnnData or file path).
                     DataFrame: cells x genes; cell IDs from the first column if it is
                     non-numeric (e.g. "pos"), otherwise from the index.
    combinations   : None = all 30; or indices 0..29 (i -> TRANSFORMATIONS[i//5] x NORMALIZATIONS[i%5])
    pipeline_genes : optional reference gene set (list / .txt / matrix file) to compare with the
                     input genes used for the library size.
    Raises StopForUser (with .log) on duplicate/unmatched IDs, NaN/Inf/negative values or
    zero library size; nothing is dropped or replaced.
    """
    say = print if verbose else (lambda *a, **k: None)
    hp = {**DEFAULT_HP, **(hp or {}), "seed": seed}
    if save_dtype not in ("float32", "float64"):
        raise ValueError("save_dtype must be float32 or float64")
    log = {"started": _now(), "package": "shiftformer", "status": "running",
           "inputs": {"train": _describe_input(train), "test": _describe_input(test),
                      "orientation": orientation}}

    def stop(reason, details):
        log.update(status="stopped_for_user", stop_reason=reason, stop_details=details)
        raise StopForUser(reason, log)

    # ---------------- alignment + checks ----------------
    rc, rg, R, _ = P.read_matrix(train, orientation)
    dc, dg, D, _ = P.read_matrix(test, orientation)
    rep = P.check_alignment(rc, rg, dc, dg)
    cperm, gperm = rep.pop("_cell_perm", None), rep.pop("_gene_perm", None)
    log["alignment"] = rep
    if not rep["ok"]:
        stop("cell/gene IDs cannot be aligned one-to-one; nothing was removed",
             {k: rep[k] for k in rep["problems"]})
    D = np.ascontiguousarray(D[np.ix_(cperm, gperm)])
    R = np.ascontiguousarray(R)
    n_cells, n_genes = R.shape
    problems = {}
    for name, M_ in (("train", R), ("test", D)):
        if (~np.isfinite(M_)).any():
            problems[f"{name}_n_nonfinite"] = int((~np.isfinite(M_)).sum())
        if (M_ < 0).any():
            problems[f"{name}_n_negative"] = int((M_ < 0).sum())
        zero = rc[P.library_size(M_) <= 0]
        if len(zero):
            problems[f"{name}_zero_library_size_cell_ids"] = [str(c) for c in zero]
    if problems:
        stop("invalid values or zero library size; nothing was removed or replaced", problems)

    # ---------------- gene set used for the library size ----------------
    gs = {"source": f"all genes of the train matrix ({_describe_input(train)})",
          "train_vs_test_gene_set": f"identical ({n_genes} genes; checked during alignment)",
          "n_genes": int(n_genes), "genes_sha256": hashlib.sha256("\n".join(rg).encode()).hexdigest(),
          "genes": [str(g) for g in rg]}
    if pipeline_genes is None:
        gs["vs_original_pipeline"] = "not_checked"
        gs["statement"] = ("Library size = sum over these input genes. NOT verified against the gene set "
                           "of the original pipeline (pass pipeline_genes to check); do not assume the "
                           "library sizes equal the original pipeline's.")
    else:
        pg = P.read_gene_ids(pipeline_genes, orientation)
        only_in, missing = sorted(set(rg) - set(pg)), sorted(set(pg) - set(rg))
        gs["original_pipeline_gene_source"] = _describe_input(pipeline_genes)
        gs["original_pipeline_n_genes"] = int(len(pg))
        if not only_in and not missing and len(pg) == len(set(pg)):
            gs["vs_original_pipeline"] = "identical"
            gs["statement"] = ("Input gene set is identical to the original pipeline's gene set, so the "
                               "library sizes follow the original definition.")
        else:
            gs["vs_original_pipeline"] = "different"
            gs.update(n_genes_only_in_input=len(only_in), n_genes_missing_from_input=len(missing),
                      genes_only_in_input_examples=only_in[:20], genes_missing_from_input_examples=missing[:20])
            gs["statement"] = (f"WARNING: input gene set differs from the original pipeline's "
                               f"({len(only_in)} extra, {len(missing)} missing). Library sizes are computed "
                               f"over the input genes and are NOT the same as the original pipeline's.")
    all_zero_ref = rg[(R == 0).all(axis=0)]
    log["data"] = {
        "n_cells_used": int(n_cells), "n_genes_used": int(n_genes),
        "test_reordered_to_train_order": {"cells": not rep["dropout_cell_order_matches_raw"],
                                          "genes": not rep["dropout_gene_order_matches_raw"]},
        "library_size_definition": P.LIBRARY_SIZE_DEFINITION, "library_size_gene_set": gs,
        "n_all_zero_reference_genes": int(len(all_zero_ref)),
        "all_zero_reference_genes": [str(g) for g in all_zero_ref],
        "n_non_integer_values": {"train": int((R != np.round(R)).sum()), "test": int((D != np.round(D)).sum())},
        "test_total_over_train_total": float(D.sum() / R.sum()),
    }

    # ---------------- shared split ----------------
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    outer = np.zeros(n_cells, dtype=int)
    roles, fold_info = {}, []
    for k, (tr, te) in enumerate(kf.split(np.arange(n_cells)), 1):
        outer[te] = k
        in_tr, in_va = train_test_split(tr, test_size=inner_val_frac, random_state=seed, shuffle=True)
        role = np.empty(n_cells, dtype=object)
        role[te], role[in_tr], role[in_va] = "outer_test", "inner_train", "inner_val"
        roles[k] = role
        fold_info.append({"fold": k, "n_outer_train": int(len(tr)), "n_inner_train": int(len(in_tr)),
                          "n_inner_val": int(len(in_va)), "n_outer_test": int(len(te)),
                          "outer_test_cell_ids": [str(c) for c in rc[te]]})
    if not ((outer >= 1).all() and sum(f["n_outer_test"] for f in fold_info) == n_cells):
        raise RuntimeError("outer folds do not partition the cells")
    splits = {"n_folds": n_folds, "outer_test_fold": outer, "inner_role": roles}
    log["split"] = {"seed": seed, "n_folds": n_folds, "inner_val_frac": inner_val_frac,
                    "outer_rule": "sklearn KFold(n_splits, shuffle=True, random_state=seed) over cells in train-matrix order",
                    "inner_rule": "sklearn train_test_split(outer_train, test_size=inner_val_frac, random_state=seed)",
                    "folds": fold_info}
    log["model"] = {"type": "TransformerAutoencoder", "hyperparameters": hp,
                    "early_stopping": {"monitor": "inner-validation MSE (transformed space)",
                                       "patience": hp["patience"], "min_delta": hp["min_delta"],
                                       "check_every": hp["check_every"], "max_epochs": hp["max_epochs"],
                                       "refit": "fresh model, same seed, all outer-train cells, best_epoch epochs"}}
    log["preprocessing"] = {
        "order": "normalization then transformation, applied to the complete train and complete test matrices; then rows taken by fold",
        "training_input": "processed train (raw) matrix, outer-train cells",
        "prediction_input": "processed test (dropout) matrix, outer-test cells",
        "inverse": "reverse transformation, then reverse normalization with the test cells' dropout library size",
        "negative_values_after_inverse": "not clipped (counted per combination)",
        "gene_or_cell_filtering": "none"}
    log["evaluation"] = {"reference": "train matrix (raw; no dropout/normalization/transformation)",
                         "correlation": "gene-wise across all cells, Pearson and Spearman (average ranks)",
                         "undefined_rule": CONSTANT_RULE, "save_dtype": save_dtype,
                         "metrics_computed_on": f"out-of-fold reconstruction cast to {save_dtype}"}
    log["fingerprint"] = {"train_sha256": _sha(R), "test_sha256": _sha(D),
                          "cells_sha256": hashlib.sha256("\n".join(rc).encode()).hexdigest(),
                          "genes_sha256": gs["genes_sha256"], "seed": seed, "n_folds": n_folds,
                          "inner_val_frac": inner_val_frac, "save_dtype": save_dtype,
                          "hp": json.dumps(hp, sort_keys=True)}
    say(f"[shiftformer] {n_cells} cells x {n_genes} genes; all-zero reference genes: {len(all_zero_ref)}; "
        f"outer test sizes: {[f['n_outer_test'] for f in fold_info]}")

    # ---------------- combinations ----------------
    combos = P.all_combos()
    if combinations is None:
        idx = list(range(len(combos)))
    else:
        idx = [int(c) for c in ([combinations] if np.isscalar(combinations) else combinations)]
        if any(i < 0 or i >= len(combos) for i in idx):
            raise ValueError("combination indices must be in 0..29")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if threads:
        torch.set_num_threads(int(threads))
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                       else ("cpu" if device == "auto" else device))
    log["device"] = str(dev)

    records, recons = {}, {}
    for i in idx:
        trans, norm = combos[i]
        t0 = time.time()
        rec = {}
        try:
            z_raw, z_drop, _, lib_drop = P.process_full(R, D, trans, norm)
            if not (np.isfinite(z_raw).all() and np.isfinite(z_drop).all()):
                raise ValueError("non-finite values after normalization/transformation")
            recon, folds = cv.run_combination(z_raw, z_drop, lib_drop, splits, trans, norm, hp, dev, log=say)
            del z_raw, z_drop
            saved = recon.astype(save_dtype)
            summ, _ = metrics.summarize(R, saved.astype(np.float64))
            rec.update(status="success", **summ,
                       best_epoch_per_fold=[f["best_epoch"] for f in folds],
                       pass1_epochs_run_per_fold=[f["pass1_epochs_run"] for f in folds],
                       folds=folds, n_negative_after_inverse=int((saved < 0).sum()),
                       checks={"each_cell_predicted_exactly_once": True,
                               "output_in_train_cell_and_gene_order": True})
            if return_reconstructions:
                recons[(trans, norm)] = pd.DataFrame(saved, index=pd.Index(rc, name="cell"), columns=rg)
        except Exception as e:  # recorded, not hidden
            rec.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        rec["runtime_sec"] = round(time.time() - t0, 1)
        records[i] = rec
        say(f"[shiftformer] {i:2d} {trans} | {norm}: {rec['status']}"
            + (f" pearson={rec['mean_pearson']!r} spearman={rec['mean_spearman']!r} "
               f"epochs={rec['best_epoch_per_fold']}" if rec["status"] == "success" else f" {rec['error']}")
            + f" ({rec['runtime_sec']}s)")

    log["combinations_requested"] = idx
    return ChooseCombinationResult(log, records, recons)
