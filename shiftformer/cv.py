"""Outer 5-fold CV for one (transformation, normalization) combination.

Inputs are the COMPLETE raw and dropout matrices already normalized + transformed
(preprocess.process_full; per-cell operations only). Per outer fold k:
  1. outer_train / outer_test cells from the shared split (identical for all combos).
  2. pass 1: inner_train / inner_val (inside outer_train, processed RAW rows only);
     early stopping on inner-val MSE -> best_epoch; model discarded.
  3. pass 2: fresh model (same seed) on all outer_train processed RAW rows for best_epoch.
  4. predict outer_test processed DROPOUT rows.
  5. reverse transformation, then reverse normalization with those cells' DROPOUT library size.
Each fit() receives a tensor holding only the rows it may see.
"""
import numpy as np
import torch

from . import model as M
from . import preprocess as P


def run_combination(z_raw, z_drop, lib_drop, splits, trans, norm, hp, device, log=print):
    n_cells, n_genes = z_raw.shape
    n_folds = int(splits["n_folds"])
    recon = np.full((n_cells, n_genes), np.nan)
    times_predicted = np.zeros(n_cells, dtype=int)
    records = []

    for k in range(1, n_folds + 1):
        role = splits["inner_role"][k]
        test_idx = np.where(role == "outer_test")[0]
        train_idx = np.where(role != "outer_test")[0]
        assert np.array_equal(test_idx, np.where(splits["outer_test_fold"] == k)[0])
        assert len(np.intersect1d(test_idx, train_idx)) == 0

        X_tr = torch.tensor(z_raw[train_idx], dtype=torch.float32)
        role_tr = role[train_idx]
        in_tr = np.where(role_tr == "inner_train")[0]
        in_va = np.where(role_tr == "inner_val")[0]

        tmp, best_ep, best_val, hist1 = M.fit(X_tr, in_tr, hp, trans, device, hp["max_epochs"], val_idx=in_va)
        del tmp
        fold_model, _, _, _ = M.fit(X_tr, np.arange(len(train_idx)), hp, trans, device, best_ep, val_idx=None)

        pred_z = M.predict(fold_model, torch.tensor(z_drop[test_idx], dtype=torch.float32),
                           hp["batch_size"], device).astype(np.float64)
        recon[test_idx] = P.inverse(pred_z, norm, trans, lib_drop[test_idx])
        times_predicted[test_idx] += 1
        del fold_model
        log(f"[{trans} | {norm}] fold {k}/{n_folds}: train={len(train_idx)} "
            f"(inner {len(in_tr)}/{len(in_va)}) test={len(test_idx)} best_epoch={best_ep} "
            f"pass1_epochs={len(hist1)} inner_val_mse={best_val:.6g}")
        records.append({"fold": k, "best_epoch": int(best_ep), "pass1_epochs_run": int(len(hist1)),
                        "early_stopped": bool(len(hist1) < hp["max_epochs"]),
                        "best_inner_val_mse": float(best_val)})

    if not np.all(times_predicted == 1):
        raise RuntimeError(f"{int((times_predicted != 1).sum())} cells not predicted exactly once")
    if not np.isfinite(recon).all():
        raise RuntimeError("reconstruction has NaN/Inf after concatenation")
    return recon, records
