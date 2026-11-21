#!/usr/bin/env python
"""Synthetic test of the shiftformer package API.
    python tests/test_synthetic.py --work_dir <scratch dir>
"""
import argparse
import copy
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import shiftformer as SF  # noqa: E402
from shiftformer import cv, metrics  # noqa: E402
from shiftformer import preprocess as P  # noqa: E402

N = 0
HP = {"n_tokens": 8, "d_model": 16, "nhead": 4, "dim_feedforward": 32, "max_epochs": 15, "patience": 3, "batch_size": 16}
KW = dict(hp=HP, device="cpu", threads=1, verbose=False)


def ok(msg):
    global N
    N += 1
    print(f"  PASS  {msg}")


def stop_details(train, test, **kw):
    try:
        SF.choose_combination(train, test, combinations=[0], **KW, **kw)
    except SF.StopForUser as e:
        return e.log["stop_details"]
    raise AssertionError("expected StopForUser")


def main(a):
    w = a.work_dir
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 40
    mu = rng.gamma(2.0, 3.0, size=n_genes)
    raw = rng.poisson(mu[None, :] * rng.uniform(0.5, 2.0, n_cells)[:, None]).astype(float)
    raw[:, 5] = 0                                              # all-zero reference gene
    drop = rng.binomial(raw.astype(int), 0.9).astype(float)    # stands in for the external dropout
    cells = np.array([f"cell_{i:03d}" for i in range(n_cells)])
    genes = np.array([f"G{j}" for j in range(n_genes)])
    train_df = pd.DataFrame(raw.copy(), index=pd.Index(cells, name="cell"), columns=genes)        # IDs in index
    cp, gp = rng.permutation(n_cells), rng.permutation(n_genes)
    test_df = pd.DataFrame(drop[np.ix_(cp, gp)], columns=genes[gp])
    test_df.insert(0, "pos", cells[cp])                                                     # IDs in "pos", shuffled

    print("[1] inputs, alignment, stop rules")
    c, g, M, _ = P.read_matrix(train_df); c2, g2, M2, _ = P.read_matrix(test_df)
    rep = P.check_alignment(c, g, c2, g2)
    assert rep["ok"] and np.array_equal(M2[np.ix_(rep["_cell_perm"], rep["_gene_perm"])], drop)
    ok("DataFrame with IDs in index / in first column; shuffled test re-ordered exactly")
    import anndata
    ad = anndata.AnnData(X=raw.astype(np.float32)); ad.obs_names = list(cells); ad.var_names = list(genes)
    c, g, M, _ = P.read_matrix(ad)
    assert list(c) == list(cells) and np.array_equal(M, raw)
    raw_f = os.path.join(w, "raw.feather"); P.write_matrix(raw_f, cells, genes, raw, "pos")
    drop_f = os.path.join(w, "drop.feather"); P.write_matrix(drop_f, cells, genes, drop, "pos")
    c, g, M, _ = P.read_matrix(raw_f)
    assert np.array_equal(M, raw)
    ok("AnnData and feather inputs read identically")
    bad = test_df.copy(); bad.loc[bad.pos == "cell_003", "pos"] = "cell_XXX"
    assert stop_details(train_df, bad) == {"cells_only_in_raw": ["cell_003"], "cells_only_in_dropout": ["cell_XXX"]}
    ok("unmatched cell IDs -> StopForUser, both sides in e.log")
    bad_csv = os.path.join(w, "bad.csv")
    g2 = genes.copy(); g2[7] = g2[8]
    with open(bad_csv, "w") as fh:
        fh.write(",".join(["pos"] + list(g2)) + "\n")
        for cc, row in zip(cells, drop):
            fh.write(",".join([cc] + [f"{v:g}" for v in row]) + "\n")
    assert stop_details(train_df, bad_csv)["dropout_duplicate_genes"] == ["G8"]
    ok("duplicate gene IDs (CSV header read verbatim) -> stop")
    d0 = pd.DataFrame(drop.copy(), index=cells, columns=genes); d0.iloc[[4, 9]] = 0
    assert stop_details(train_df, d0) == {"test_zero_library_size_cell_ids": ["cell_004", "cell_009"]}
    ok("zero library size -> stop, cell IDs reported")
    try:
        P.apply_norm(raw, "standardize", raw.sum(1)); raise AssertionError
    except NotImplementedError:
        ok("cross-cell normalization refused")

    print("[2] preprocessing / inverse / correlation")
    for t, n in P.all_combos():
        zr, zd, lr, ld = P.process_full(raw, drop, t, n)
        assert np.allclose(P.inverse(zr, n, t, lr), raw, atol=1e-8) and np.allclose(P.inverse(zd, n, t, ld), drop, atol=1e-8)
        sub = np.array([1, 7, 30])
        assert np.array_equal(P.apply_trans(P.apply_norm(raw[sub], n, raw[sub].sum(1)), t), zr[sub])
    ok("30 combos invertible with own library sizes; full-matrix processing == per-subset")
    X = rng.normal(size=(30, 6)); Y = X + rng.normal(size=(30, 6)); X[:, 2] = 3.0; Y[:, 4] = 0.0
    for m, f in (("pearson", pearsonr), ("spearman", spearmanr)):
        r, u = metrics.per_gene_correlation(X, Y, m)
        assert u.tolist() == [False, False, True, False, True, False] and r[2] == 0 and r[4] == 0
        assert all(np.isclose(r[j], f(X[:, j], Y[:, j])[0], atol=1e-12) for j in (0, 1, 3, 5))
    ok("correlations match scipy; constant -> undefined -> 0")

    print("[3] leakage (fold internals)")
    from sklearn.model_selection import KFold, train_test_split
    outer = np.zeros(n_cells, int); roles = {}
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(np.arange(n_cells)), 1):
        outer[te] = k; i_tr, i_va = train_test_split(tr, test_size=0.1, random_state=42)
        rr = np.empty(n_cells, object); rr[te], rr[i_tr], rr[i_va] = "outer_test", "inner_train", "inner_val"; roles[k] = rr
    splits = {"n_folds": 5, "outer_test_fold": outer, "inner_role": roles}
    hp = {**SF.pipeline.DEFAULT_HP, **HP, "seed": 42}; dev = torch.device("cpu")

    def go(Rm, Dm):
        zr, zd, _, ld = P.process_full(Rm, Dm, "log2", "10000")
        return cv.run_combination(zr, zd, ld, splits, "log2", "10000", hp, dev, log=lambda *x: None)
    base, f0 = go(raw, drop)
    assert np.array_equal(base, go(raw, drop)[0]); ok("bit-identical on rerun")
    t1 = outer == 1
    R2 = raw.copy(); R2[t1] = R2[t1] * 7 + 3; rec, f2 = go(R2, drop)
    assert np.array_equal(rec[t1], base[t1]) and f2[0]["best_epoch"] == f0[0]["best_epoch"] and not np.allclose(rec[~t1], base[~t1])
    ok("outer-test raw unused by its fold, used by the others")
    D2 = drop.copy(); D2[~t1] = D2[~t1] * 5 + 1
    assert np.array_equal(go(raw, D2)[0][t1], base[t1]); ok("dropout of training cells unused")
    D3 = drop.copy(); D3[t1] = D3[t1] * 3 + 2; rec = go(raw, D3)[0]
    assert np.array_equal(rec[~t1], base[~t1]) and not np.allclose(rec[t1], base[t1]); ok("dropout only predicts its own test cells")

    print("[4] full API run")
    res = SF.choose_combination(train_df, test_df, return_reconstructions=True, **KW)
    assert res.final and res.n_successful == 30 and list(res.summary.columns) == SF.pipeline.SUMMARY_COLUMNS
    rc = res.reconstructions[("log2", "10000")]
    assert list(rc.index) == list(cells) and list(rc.columns) == list(genes)
    assert np.array_equal(rc.to_numpy(), base.astype(np.float32))
    ok("choose_combination: 30/30; optional reconstructions in train cell/gene order == fold internals")
    row = res.summary[(res.summary.transformation == "log2") & (res.summary.normalization == "10000")].iloc[0]
    assert row.mean_pearson == metrics.summarize(raw, base.astype(np.float32).astype(np.float64))[0]["mean_pearson"]
    assert (res.summary.n_undefined_pearson >= 1).all()
    ok("metrics computed on the float32 reconstruction; all-zero gene counted undefined")
    out = res.save(os.path.join(w, "full"))
    assert sorted(os.listdir(out)) == ["report.txt", "run_log.json", "summary.tsv"]
    sm = pd.read_csv(os.path.join(out, "summary.tsv"), sep="\t", float_precision="round_trip")
    assert sm.mean_pearson.max() == res.best_by_pearson["score"] and "FINAL best" in res.report
    lg = res.log
    assert lg["split"]["seed"] == 42 and sum(len(f["outer_test_cell_ids"]) for f in lg["split"]["folds"]) == 60
    assert all(len(c["best_epoch_per_fold"]) == 5 for c in lg["combinations"]) and lg["model"]["early_stopping"]["min_delta"] == 0.001
    assert lg["evaluation"]["save_dtype"] == "float32" and lg["data"]["n_all_zero_reference_genes"] == 1
    assert lg["data"]["library_size_gene_set"]["vs_original_pipeline"] == "not_checked" and "NOT verified" in res.report
    ok("save(): 3 files, exact float64 round trip; run_log has seed, folds, epochs, ES, dtype, gene set")
    re_ = SF.load_result(out)
    assert re_.summary.equals(res.summary) and re_.report == res.report
    ok("load_result() reproduces summary and report")

    print("[5] partial runs, failures, merge, ties")
    p1 = SF.choose_combination(train_df, test_df, combinations=range(0, 28), **KW)
    assert not p1.final and p1.summary.mean_pearson.isna().sum() == 2
    assert "PROVISIONAL" in p1.report and "FINAL" not in p1.report and "not_run" in p1.report
    ok("28/30 -> NA rows, not_run listed, PROVISIONAL (never FINAL)")
    p1.save(os.path.join(w, "part1"))
    p2 = SF.choose_combination(raw_f, drop_f, combinations=[28, 29], **KW); p2.save(os.path.join(w, "part2"))
    mg = SF.merge_results([SF.load_result(os.path.join(w, "part1")), SF.load_result(os.path.join(w, "part2"))])
    assert mg.final and mg.summary.mean_pearson.equals(res.summary.mean_pearson)
    ok("parts (DataFrame input + file input) merge to the same 30 scores as one full run -> FINAL")
    p3 = SF.choose_combination(raw_f, drop_f, combinations=[29], seed=7, **KW)
    try:
        SF.merge_results([p1, p3]); raise AssertionError("merge accepted different settings")
    except ValueError:
        ok("merge refuses parts computed with different settings (seed)")
    env = {**os.environ, "PYTHONPATH": ROOT}
    r = subprocess.run([sys.executable, "-m", "shiftformer", "merge", "--parts", os.path.join(w, "part1"),
                        os.path.join(w, "part2"), "--out_dir", os.path.join(w, "merged_cli")], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "FINAL best" in open(os.path.join(w, "merged_cli", "report.txt")).read()
    r = subprocess.run([sys.executable, "-m", "shiftformer", "choose", "--train", raw_f, "--test", os.path.join(w, "bad.csv"),
                        "--out_dir", os.path.join(w, "cli_stop")], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and os.path.exists(os.path.join(w, "cli_stop", "run_log.json"))
    ok("CLI merge works; CLI choose stops with exit 2 and writes run_log.json")
    recs = copy.deepcopy(res.records)
    recs[3] = {"status": "failed", "error": "ValueError: simulated failure"}
    fr = SF.ChooseCombinationResult(copy.deepcopy(res.log), recs)
    assert "simulated failure" in fr.report and "PROVISIONAL" in fr.report and not fr.final
    ok("failed combination -> NA, reason in report, PROVISIONAL")
    recs = copy.deepcopy(res.records)
    lead = max(recs, key=lambda i: recs[i]["mean_pearson"]); other = (lead + 1) % 30; third = (lead + 2) % 30
    recs[other]["mean_pearson"] = recs[lead]["mean_pearson"]
    recs[third]["mean_pearson"] = float(np.nextafter(recs[lead]["mean_pearson"], 0))
    tr = SF.ChooseCombinationResult(copy.deepcopy(res.log), recs)
    assert len(tr.best_by_pearson["combinations"]) == 2 and "[2 combinations exactly tied]" in tr.report
    ok("exact float64 tie -> both listed; 1 ulp lower is not a tie")

    print("[6] reference gene set")
    for ref, want in ((list(genes[::-1]), "identical"), (raw_f, "identical"), (list(genes[:30]) + ["GX"], "different")):
        rr = SF.choose_combination(train_df, test_df, combinations=[0], pipeline_genes=ref, **KW)
        g = rr.log["data"]["library_size_gene_set"]
        assert g["vs_original_pipeline"] == want
    assert g["n_genes_only_in_input"] == 10 and g["n_genes_missing_from_input"] == 1 and g["statement"].startswith("WARNING")
    ok("pipeline_genes: identical (list / matrix file) or different with WARNING + counts")
    print(res.report)
    print(f"ALL {N} CHECKS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    main(ap.parse_args())
