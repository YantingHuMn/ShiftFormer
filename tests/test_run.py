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
    # default: log2(count+2); rule picks 1000 (pearson) and 1000000 (spearman) on this synthetic matrix
    fp = "pearson_reconstruct_trans_by_log2(count+2)_norm_by_1000.feather"
    fs = "spearman_reconstruct_trans_by_log2(count+2)_norm_by_1000000.feather"
    assert {fp, fs, "config.json", "report.txt", "evaluation.tsv",
            "model_output_pearson.feather", "model_output_spearman.feather"} == files, files
    assert [r1.config["outputs"][c]["file"] for c in ("pearson", "spearman")] == [fp, fs]
    assert f"file={fp}" in r1.report and f"file={fs}" in r1.report
    assert json.load(open(os.path.join(out, "config.json")))["outputs"]["pearson"]["file"] == fp
    for crit, f in (("pearson", fp), ("spearman", fs)):
        x = pd.read_feather(os.path.join(out, f))
        assert list(x.columns) == list(df.columns) and x.pos.tolist() == df.pos.tolist()
        assert (x.iloc[:, 1:].to_numpy() >= 0).all()
        assert np.array_equal(x.iloc[:, 1:].to_numpy(), r1.reconstructions[crit].to_numpy().astype("float32"))
    out_b = os.path.join(w, "run_save")
    r1.save(out_b)                                                     # result.save == run(out_dir=...)
    assert set(os.listdir(out_b)) == files
    for f in (fp, fs):
        assert pd.read_feather(os.path.join(out, f)).equals(pd.read_feather(os.path.join(out_b, f)))
    print("  PASS  outputs named by actual trans/norm (run(out_dir) == result.save; config/report show names)")

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
    out3 = os.path.join(w, "run_shared")
    r3.save(out3)
    f3 = [f"{c}_reconstruct_trans_by_log2(count+2)_norm_by_1000000.feather" for c in ("pearson", "spearman")]
    assert set(f3) <= set(os.listdir(out3))
    assert pd.read_feather(os.path.join(out3, f3[0])).equals(pd.read_feather(os.path.join(out3, f3[1])))
    print("  PASS  shared model -> still two files (pearson_ and spearman_)")

    out4 = os.path.join(w, "run_manual")
    r4 = SF.run(df, out_dir=out4, transformation="sqrt+1", pearson_norm="no_norm", spearman_norm="10000",
                hp={**HP, "max_epochs": 2}, threads=1, verbose=False)
    f4 = ["pearson_reconstruct_trans_by_sqrt+1_norm_by_no_norm.feather",
          "spearman_reconstruct_trans_by_sqrt+1_norm_by_10000.feather"]
    assert [r4.config["outputs"][c]["file"] for c in ("pearson", "spearman")] == f4
    assert set(f4) <= set(os.listdir(out4)) and all(f"file={f}" in r4.report for f in f4)
    print("  PASS  user-given transformation / norms (incl. no_norm) used in the file names")

    from shiftformer.workflow import output_filename, safe_filename_part
    assert safe_filename_part("log2(count+2)") == "log2(count+2)"
    assert safe_filename_part('a/b\\c:d*e?f"g<h>i|j k') == "a_b_c_d_e_f_g_h_i_j_k"
    assert output_filename("pearson", "x/y", "1e4") == "pearson_reconstruct_trans_by_x_y_norm_by_1e4.feather"
    print("  PASS  unsafe file-name characters -> '_' ( ( ) + kept )")

    bad = df.copy()
    bad.iloc[0, 1:] = 0
    try:
        SF.run(bad, hp=HP, verbose=False)
        raise AssertionError("expected StopForUser")
    except SF.StopForUser as e:
        assert "zero_library_size_cell_ids" in e.log["stop_details"]
    print("  PASS  zero library size -> StopForUser")
    json.load(open(os.path.join(out, "config.json")))

    # ---------------- transformation parameter ----------------
    assert r1.config["transformation"] == rule.TRANSFORMATION == "log2(count+2)"
    assert "transformation: log2(count+2)" in r1.report and "NOTE" not in r1.report
    r_def = SF.run(df, hp=HP, threads=1, save_model_space=True, verbose=False, transformation="log2(count+2)")
    for crit in ("pearson", "spearman"):
        assert np.array_equal(r_def.model_space[crit].to_numpy(), r1.model_space[crit].to_numpy())
        assert np.array_equal(r_def.reconstructions[crit].to_numpy(), r1.reconstructions[crit].to_numpy())
    print("  PASS  default transformation = explicit log2(count+2) (identical output)")

    # example call from the request: custom transformation + both norms given
    r_l2 = SF.run(df, transformation="log2", pearson_norm="10000", spearman_norm="10000",
                  hp=HP, threads=1, save_model_space=True, verbose=False)
    c = r_l2.config
    assert c["transformation"] == "log2" and c["rule"]["transformation"] == "log2"
    assert [c["outputs"][k]["normalization"] for k in ("pearson", "spearman")] == ["10000", "10000"]
    assert [c["outputs"][k]["normalization_source"] for k in ("pearson", "spearman")] == ["user", "user"]
    assert c["outputs"]["pearson"]["shared_model_with"] == ["spearman"]
    assert "transformation: log2 " in r_l2.report and "NOTE" not in r_l2.report
    zl2, _, _ = train_one_norm(raw, "10000", "log2", hp, torch.device("cpu"), 42, 0.1, say=lambda *a: None)
    assert np.array_equal(zl2, r_l2.model_space["pearson"].to_numpy())                      # forward + training
    lib = raw.sum(1)
    inv = np.maximum((2.0 ** zl2.astype(np.float64) - 1) / 10000 * lib[:, None], 0)     # log2 inverse: 2^y - 1
    assert np.array_equal(inv, r_l2.reconstructions["pearson"].to_numpy())
    z_l2p2, _, _ = train_one_norm(raw, "10000", rule.TRANSFORMATION, hp, torch.device("cpu"), 42, 0.1,
                                  say=lambda *a: None)
    assert not np.array_equal(zl2, z_l2p2)
    print("  PASS  transformation='log2' used in forward, training and inverse; recorded in config/report")

    for t in SF.TRANSFORMATIONS:
        rt = SF.run(df, transformation=t, pearson_norm="1000", spearman_norm="no_norm", hp={**HP, "max_epochs": 2},
                    threads=1, verbose=False)
        assert rt.config["transformation"] == t and f"transformation: {t} " in rt.report
        assert all(np.isfinite(v.to_numpy()).all() for v in rt.reconstructions.values())
    print(f"  PASS  all {len(SF.TRANSFORMATIONS)} transformations run: {SF.TRANSFORMATIONS}")

    r_note = SF.run(df, transformation="sqrt", hp={**HP, "max_epochs": 2}, threads=1, verbose=False)
    assert "fitted for log2(count+2)" in r_note.config["rule"]["note"] and "NOTE" in r_note.report
    print("  PASS  rule-chosen norm with a non-default transformation -> note in config/report")

    for badt in ("log10", "LOG2", "", None):
        try:
            SF.run(df, transformation=badt, hp=HP, verbose=False)
            raise AssertionError(f"expected ValueError for {badt!r}")
        except ValueError as e:
            assert "unsupported transformation" in str(e) and "log2(count+2)" in str(e)
    print("  PASS  unsupported transformation -> clear ValueError")

    # command line
    import subprocess
    csv = os.path.join(w, "tiny.feather")
    df.to_feather(csv)
    env = {**os.environ, "PYTHONPATH": ROOT, "PYTHONDONTWRITEBYTECODE": "1"}
    base = [sys.executable, "-m", "shiftformer", "run", "--input", csv, "--threads", "1"]
    p = subprocess.run(base + ["--out_dir", os.path.join(w, "cli_bad"), "--transformation", "log10"],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 2 and "invalid choice" in p.stderr, p.stderr
    p = subprocess.run(base + ["--out_dir", os.path.join(w, "cli_log2"), "--transformation", "log2",
                               "--pearson_norm", "10000", "--spearman_norm", "10000"],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert json.load(open(os.path.join(w, "cli_log2", "config.json")))["transformation"] == "log2"
    print("  PASS  CLI --transformation (valid value used, invalid value rejected)")
    print("all tests passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    main(ap.parse_args())
