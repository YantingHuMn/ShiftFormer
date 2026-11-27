#!/usr/bin/env python
"""Synthetic test of the default workflow (shiftformer.run / shiftformer.choose_norm).
    python tests/test_run.py --work_dir <scratch dir>
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import shiftformer as SF  # noqa: E402
from shiftformer import rule  # noqa: E402
from shiftformer.workflow import train_one_norm  # noqa: E402

HP = {"n_tokens": 8, "d_model": 16, "nhead": 4, "dim_feedforward": 32, "max_epochs": 6, "patience": 3,
      "batch_size": 16}


def feats(det, lib):
    return dict(median_detected_genes=det, median_library_size=lib, n_cells=0, n_genes=0)


def main(a):
    w = a.work_dir
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    torch.set_num_threads(1)

    # rule: values checked against the fitted choices of the original analysis
    assert SF.choose_norm(features=feats(1344, 2500))["pearson"]["normalization"] == "10000"     # 4.491 -> 1e4
    assert SF.choose_norm(features=feats(8064, 20000))["pearson"]["normalization"] == "100000"   # 5.099 -> 1e5
    assert SF.choose_norm(features=feats(621, 826))["pearson"]["normalization"] == "10000"       # 4.229 -> 1e4
    assert SF.choose_norm(features=feats(100, 1e9))["spearman"]["normalization"] == rule.SPEARMAN_NORM
    print("  PASS  rule")

    rng = np.random.default_rng(0)
    n, g = 80, 40
    raw = rng.poisson(rng.gamma(2, 3, g)[None, :] * rng.uniform(.5, 2, n)[:, None]).astype(float)
    df = pd.DataFrame(raw, columns=[f"g{i}" for i in range(g)])
    df.insert(0, "pos", [f"c{i}" for i in range(n)])
    ref = df.copy()
    ref.iloc[:, 1:] = rng.poisson(raw + 1)

    out = os.path.join(w, "run")
    r1 = SF.run(df, reference=ref, out_dir=out, hp=HP, threads=1, save_model_space=True, verbose=False)
    files = set(os.listdir(out))
    assert {"recon_pearson.feather", "recon_spearman.feather", "config.json", "report.txt",
            "evaluation.tsv"} <= files, files
    assert not any(f.startswith("summary") for f in files)            # 30-combination summary not written
    for crit in ("pearson", "spearman"):
        x = pd.read_feather(os.path.join(out, f"recon_{crit}.feather"))
        assert list(x.columns) == list(df.columns) and x.pos.tolist() == df.pos.tolist()
        assert (x.iloc[:, 1:].to_numpy() >= 0).all()
    print("  PASS  outputs (two recon files, same cells/genes order, clipped at 0)")

    r2 = SF.run(df, hp=HP, threads=1, save_model_space=True, verbose=False)
    for crit in ("pearson", "spearman"):
        assert np.array_equal(r1.model_space[crit].to_numpy(), r2.model_space[crit].to_numpy())
    print("  PASS  deterministic (same seed -> identical output)")

    # the run must equal a hand-written copy of the original two-pass procedure
    norm = r1.config["outputs"]["pearson"]["normalization"]
    hp = {**SF.pipeline.DEFAULT_HP, **HP, "seed": 42}
    zhat, _, info = train_one_norm(raw, norm, rule.TRANSFORMATION, hp, torch.device("cpu"), 42, 0.1,
                                                 say=lambda *a: None)
    assert np.array_equal(zhat, r1.model_space["pearson"].to_numpy())
    assert info["best_epoch"] == r1.config["outputs"]["pearson"]["best_epoch"]
    print("  PASS  train_one_norm == run")

    r3 = SF.run(df, hp=HP, threads=1, verbose=False, pearson_norm="1000000", clip_negative=False)
    assert r3.config["outputs"]["pearson"]["shared_model_with"] == ["spearman"]
    assert np.array_equal(r3.reconstructions["pearson"].to_numpy(), r3.reconstructions["spearman"].to_numpy())
    print("  PASS  same norm for both criteria -> trained once")

    bad = df.copy()
    bad.iloc[0, 1:] = 0
    try:
        SF.run(bad, hp=HP, verbose=False)
        raise AssertionError("expected StopForUser")
    except SF.StopForUser as e:
        assert "zero_library_size_cell_ids" in e.log["stop_details"]
    print("  PASS  zero library size -> StopForUser")
    json.load(open(os.path.join(out, "config.json")))
    print("all tests passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    main(ap.parse_args())
