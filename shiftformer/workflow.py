"""Default ShiftFormer workflow: fixed transformation + rule-chosen normalization.

    res = shiftformer.run(counts)                       # raw count matrix, cells x genes
    res.reconstructions["pearson"], res.reconstructions["spearman"]   # count-scale DataFrames
    res.config, res.report
    res.save(out_dir)   # recon_pearson.feather, recon_spearman.feather, config.json, report.txt

1. transformation fixed to log2(count+2)
2. normalization chosen from the input alone (shiftformer.rule):
     pearson : log10 N = 2.048 + 0.781 * log10(median detected genes), snapped to 1e3..1e6/no_norm
     spearman: N = 1e6
3. for each chosen normalization (trained once if both criteria pick the same N) the training is
   the one of the original One_Shifting Transformer_denoise run (Step2_train_Transformer_denoise.py
   with a single configuration + Step3_reconstruct_Transformer_denoise.py), no dropout:
     forward  : x / lib_size * N  (per cell; skipped for no_norm), then log2(x + 2)
     pass 1   : train_test_split(all cells, test_size=0.1, random_state=seed); early stopping on the
                validation MSE (patience 10, min_delta 0.001, max 100 epochs) -> best_epoch
     final    : fresh model (same seed) trained on ALL cells for best_epoch epochs, no validation
     recon    : final model applied to all cells (eval mode, batches of 64)
     inverse  : 2^y - 2, then / N * lib_size; negative values clipped to 0 (clip_negative=True)
4. optional: if a reference matrix is given, gene-wise Pearson / Spearman of each output vs the
   reference (and of the raw input vs the reference) are reported. The reference is never used
   for training or for choosing anything.

The 30-combination search (choose_combination) is still available but is not run by default.
"""
import datetime as _dt
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from . import metrics
from . import model as M
from . import preprocess as P
from . import rule as RULE
from .pipeline import DEFAULT_HP, StopForUser

CRITERIA = ("pearson", "spearman")


def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def _describe(x):
    return str(x) if isinstance(x, (str, os.PathLike)) else f"<{type(x).__name__}>"


class RunResult:
    """reconstructions {criterion: DataFrame cells x genes (count scale)}, config, evaluation, report."""

    def __init__(self, config, reconstructions, id_col, model_space=None, evaluation=None):
        self.config = config
        self.reconstructions = reconstructions
        self.model_space = model_space or {}
        self.evaluation = evaluation
        self._id_col = id_col
        self.report = self._make_report()

    def _make_report(self):
        c = self.config
        f = c["rule"]["features"]
        lines = ["ShiftFormer (Transformer denoise, rule-chosen normalization)",
                 f"cells={f['n_cells']}  genes={f['n_genes']}  median detected genes={f['median_detected_genes']:g}  "
                 f"median library size={f['median_library_size']:g}",
                 f"transformation: {c['transformation']}  seed={c['seed']}  device={c['device']}"]
        for crit in CRITERIA:
            o = c["outputs"][crit]
            lines.append(f"{crit:8s}: normalization={o['normalization']} ({o['normalization_source']})  "
                         f"best_epoch={o['best_epoch']}  negatives after inverse={o['n_negative_after_inverse']}"
                         f"{' (clipped to 0)' if c['clip_negative'] else ''}  file={o['file']}")
        if c["rule"]["pearson"].get("predicted_log10N") is not None:
            lines.append(f"pearson rule: predicted log10 N = {c['rule']['pearson']['predicted_log10N']:.3f}")
        if self.evaluation is not None:
            lines.append("evaluation vs reference (mean gene-wise correlation; undefined -> 0):")
            lines.append(self.evaluation.to_string(index=False))
        return "\n".join(lines) + "\n"

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        dt = self.config["save_dtype"]
        for crit, df in self.reconstructions.items():
            P.write_matrix(os.path.join(out_dir, self.config["outputs"][crit]["file"]), df.index.to_numpy(),
                           df.columns.to_numpy(), df.to_numpy().astype(dt), self._id_col)
        for crit, df in self.model_space.items():
            P.write_matrix(os.path.join(out_dir, f"model_output_{crit}.feather"), df.index.to_numpy(),
                           df.columns.to_numpy(), df.to_numpy(), self._id_col)
        if self.evaluation is not None:
            self.evaluation.to_csv(os.path.join(out_dir, "evaluation.tsv"), sep="\t", index=False,
                                   float_format="%.17g")
        with open(os.path.join(out_dir, "config.json"), "w") as fh:
            json.dump(self.config, fh, indent=2, default=str)
        with open(os.path.join(out_dir, "report.txt"), "w") as fh:
            fh.write(self.report)
        return out_dir

    def __repr__(self):
        return self.report


def _check_input(cells, genes, R):
    problems = {}
    for what, ids in (("cell", cells), ("gene", genes)):
        s = pd.Series(ids)
        d = sorted(s[s.duplicated(keep=False)].unique().tolist())
        if d:
            problems[f"duplicate_{what}_ids"] = d[:20]
    if (~np.isfinite(R)).any():
        problems["n_nonfinite"] = int((~np.isfinite(R)).sum())
    elif (R < 0).any():
        problems["n_negative"] = int((R < 0).sum())
    else:
        zero = cells[P.library_size(R) <= 0]
        if len(zero):
            problems["zero_library_size_cell_ids"] = [str(c) for c in zero[:20]]
            problems["n_zero_library_size_cells"] = int(len(zero))
    return problems


def train_one_norm(R, norm, trans, hp, device, seed, inner_val_frac, say=print):
    """Original Step2 (single configuration) + Step3 for one normalization.
    Returns (model-space output float32, count-scale reconstruction float64 (unclipped), info)."""
    z, lib = P.forward(R, norm, trans)
    X = torch.tensor(z, dtype=torch.float32)
    del z
    if not torch.isfinite(X).all():
        raise ValueError(f"non-finite values after normalization {norm} + {trans}")
    n = X.shape[0]
    all_idx = np.arange(n)
    tr, va = train_test_split(all_idx, test_size=inner_val_frac, random_state=seed, shuffle=True)
    t0 = time.time()
    tmp, best_ep, best_val, hist = M.fit(X, tr, hp, trans, device, hp["max_epochs"], val_idx=va)
    del tmp                                   # pass-1 model only supplies best_epoch
    say(f"[shiftformer] norm={norm}: pass 1 ({len(tr)} train / {len(va)} val cells) "
        f"best_epoch={best_ep} val_mse={best_val:.6g} epochs_run={len(hist)}")
    final, _, _, _ = M.fit(X, all_idx, hp, trans, device, best_ep, val_idx=None)
    zhat = M.predict(final, X, hp["batch_size"], device)            # float32, like Step3
    del final, X
    recon = P.inverse(zhat.astype(np.float64), norm, trans, lib)
    info = {"best_epoch": int(best_ep), "pass1_epochs_run": int(len(hist)),
            "early_stopped": bool(len(hist) < hp["max_epochs"]), "best_inner_val_mse": float(best_val),
            "n_pass1_train_cells": int(len(tr)), "n_pass1_val_cells": int(len(va)),
            "n_final_train_cells": int(n), "train_seconds": round(time.time() - t0, 1)}
    say(f"[shiftformer] norm={norm}: final model on all {n} cells for {best_ep} epochs "
        f"({info['train_seconds']}s)")
    return zhat, recon, info


def run(counts, reference=None, out_dir=None, orientation="cells_x_genes", seed=42,
        pearson_norm=None, spearman_norm=None, clip_negative=True, save_dtype="float32",
        save_model_space=False, inner_val_frac=0.1, device="cpu", threads=None, hp=None,
        verbose=True):
    """Denoise `counts` with Transformer denoise, log2(count+2) and the rule-chosen normalization.

    counts        : raw count matrix (no normalization / transformation). DataFrame (cells x genes;
                    cell IDs in the first non-numeric column, e.g. "pos", or the index), AnnData, or a
                    file (.feather/.parquet/.csv/.tsv[.gz]).
    reference     : optional matrix of the same cells (and genes) used ONLY for evaluation.
    out_dir       : if given, results are saved there (see RunResult.save).
    pearson_norm / spearman_norm : override the rule (one of 1000, 10000, 100000, 1000000, no_norm).
    clip_negative : clip negative values after the inverse transformation to 0 (default True).
    save_model_space : also keep the raw model output (log2 space, float32), as written by the
                    original Step3 (reconstruct_trans_by_..._norm_by_....feather).
    device        : "cpu" (default; the original runs used CPU), "cuda" or "auto".
    threads       : torch CPU threads (the original runs used 1).
    hp            : override Transformer hyperparameters (default = original Transformer_denoise settings).
    """
    say = print if verbose else (lambda *a, **k: None)
    trans = RULE.TRANSFORMATION
    hp = {**DEFAULT_HP, **(hp or {}), "seed": seed}
    if save_dtype not in ("float32", "float64"):
        raise ValueError("save_dtype must be float32 or float64")
    cfg = {"started": _now(), "package": "shiftformer", "workflow": "rule_chosen_normalization",
           "input": _describe(counts), "orientation": orientation, "transformation": trans,
           "seed": seed, "clip_negative": clip_negative, "save_dtype": save_dtype}

    cells, genes, R, id_col = P.read_matrix(counts, orientation)
    problems = _check_input(cells, genes, R)
    if problems:
        cfg.update(status="stopped_for_user", stop_details=problems)
        raise StopForUser("invalid input (duplicate IDs, NaN/Inf, negative values or zero library size); "
                          "nothing was removed or replaced", cfg)
    n_zero_genes = int((R == 0).all(axis=0).sum())
    cfg["data"] = {"n_cells": int(R.shape[0]), "n_genes": int(R.shape[1]),
                   "n_all_zero_genes": n_zero_genes,
                   "library_size_definition": "per-cell sum of counts over all genes of the input matrix",
                   "note_all_zero_genes": ("kept; the original pipeline input had none (genes detected in < 2 "
                                           "cells were removed when the matrices were built)")
                   if n_zero_genes else None}

    # ---------------- rule ----------------
    choice = RULE.choose_norm(R)
    sources = {}
    for crit, override in (("pearson", pearson_norm), ("spearman", spearman_norm)):
        if override is None:
            sources[crit] = "rule"
        else:
            override = str(override)
            if override not in RULE.NORM_OPTIONS:
                raise ValueError(f"{crit}_norm must be one of {RULE.NORM_OPTIONS}")
            choice[crit]["normalization"] = override
            sources[crit] = "user"
    cfg["rule"] = choice
    say(f"[shiftformer] {R.shape[0]} cells x {R.shape[1]} genes; median detected genes="
        f"{choice['features']['median_detected_genes']:g}, median library size="
        f"{choice['features']['median_library_size']:g}")
    say(f"[shiftformer] pearson -> {choice['pearson']['normalization']} ({sources['pearson']}), "
        f"spearman -> {choice['spearman']['normalization']} ({sources['spearman']})")

    # ---------------- training ----------------
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if threads:
        torch.set_num_threads(int(threads))
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                       else ("cpu" if device == "auto" else device))
    cfg.update(device=str(dev), torch_threads=torch.get_num_threads(),
               versions={"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
               model={"type": "TransformerAutoencoder", "hyperparameters": hp,
                      "procedure": ("pass 1: train_test_split(all cells, test_size=inner_val_frac, "
                                    "random_state=seed), early stopping on validation MSE -> best_epoch; "
                                    "final: fresh model (same seed) on all cells for best_epoch epochs; "
                                    "reconstruction: final model on all cells"),
                      "inner_val_frac": inner_val_frac})

    trained, recons, zspace, outputs = {}, {}, {}, {}
    for crit in CRITERIA:
        norm = choice[crit]["normalization"]
        if norm not in trained:
            trained[norm] = train_one_norm(R, norm, trans, hp, dev, seed, inner_val_frac, say)
        zhat, recon, info = trained[norm]
        n_neg = int((recon < 0).sum())
        out = np.maximum(recon, 0) if clip_negative else recon.copy()
        recons[crit] = pd.DataFrame(out, index=pd.Index(cells, name=id_col), columns=genes)
        if save_model_space:
            zspace[crit] = pd.DataFrame(zhat, index=pd.Index(cells, name=id_col), columns=genes)
        outputs[crit] = {"normalization": norm, "normalization_source": sources[crit], **info,
                         "shared_model_with": [c for c in CRITERIA if c != crit
                                               and choice[c]["normalization"] == norm],
                         "n_negative_after_inverse": n_neg,
                         "frac_negative_after_inverse": n_neg / recon.size,
                         "file": f"recon_{crit}.feather"}
    del trained
    cfg["outputs"] = outputs

    # ---------------- optional evaluation ----------------
    evaluation = None
    if reference is not None:
        evaluation, cfg["evaluation"] = _evaluate(reference, orientation, cells, genes, R, recons,
                                                   {c: outputs[c]["normalization"] for c in CRITERIA})
    cfg.update(status="complete", finished=_now())
    res = RunResult(cfg, recons, id_col, zspace, evaluation)
    if out_dir is not None:
        res.save(out_dir)
        say(f"[shiftformer] saved to {out_dir}")
    say(res.report)
    return res


def _evaluate(reference, orientation, cells, genes, R, recons, norms):
    rc, rg, REF, _ = P.read_matrix(reference, orientation)
    missing_cells = sorted(set(cells) - set(rc))
    if missing_cells:
        raise ValueError(f"{len(missing_cells)} input cells are missing from the reference, "
                         f"e.g. {missing_cells[:5]}")
    gpos = {g: i for i, g in enumerate(rg)}
    keep = np.array([g in gpos for g in genes])
    cpos = {c: i for i, c in enumerate(rc)}
    REF = REF[np.ix_([cpos[c] for c in cells], [gpos[g] for g in genes[keep]])]
    rows = [dict(output="input (raw)", normalization="", **metrics.summarize(REF, R[:, keep])[0])]
    for crit, df in recons.items():
        rows.append(dict(output=f"recon_{crit}", normalization=norms[crit],
                         **metrics.summarize(REF, df.to_numpy()[:, keep])[0]))
    ev = pd.DataFrame(rows)
    info = {"reference": _describe(reference), "n_genes_evaluated": int(keep.sum()),
            "n_input_genes_missing_from_reference": int((~keep).sum()),
            "correlation": "gene-wise across cells; undefined (constant vector) -> 0; mean over genes",
            "note": "reference used only for this evaluation, never for training or choosing"}
    return ev, info
