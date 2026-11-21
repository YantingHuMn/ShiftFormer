"""Per-gene (column-wise, across cells) Pearson / Spearman correlation.

A gene's correlation is UNDEFINED when either vector is constant across cells
(max == min, tested exactly, not via a floating-point SD) or there are < 2 cells.
Undefined correlations are recorded as 0 and counted; the mean is over ALL genes.
Spearman = Pearson on average ranks (same as scipy.stats.spearmanr, R cor(method="spearman")).
"""
import numpy as np
from scipy.stats import rankdata


def _pearson_cols(X, Y):
    xc = X - X.mean(axis=0)
    yc = Y - Y.mean(axis=0)
    num = (xc * yc).sum(axis=0)
    den = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.clip(r, -1.0, 1.0)


def per_gene_correlation(truth, pred, method, chunk=2000):
    # C-contiguous float64 copies + contiguous chunks: summation order (and therefore every bit of
    # the result) does not depend on the caller's memory layout -> exact ties are meaningful.
    truth = np.ascontiguousarray(truth, dtype=np.float64)
    pred = np.ascontiguousarray(pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch {truth.shape} vs {pred.shape}")
    n_cells, n_genes = truth.shape
    r = np.zeros(n_genes)
    undefined = np.zeros(n_genes, dtype=bool)
    for s in range(0, n_genes, chunk):
        X = np.ascontiguousarray(truth[:, s:s + chunk])
        Y = np.ascontiguousarray(pred[:, s:s + chunk])
        undef = (np.ptp(X, axis=0) == 0) | (np.ptp(Y, axis=0) == 0) | (n_cells < 2)
        if method == "spearman":
            X = rankdata(X, axis=0, method="average")
            Y = rankdata(Y, axis=0, method="average")
        elif method != "pearson":
            raise ValueError(method)
        rr = _pearson_cols(X, Y)
        undef |= ~np.isfinite(rr)
        rr[undef] = 0.0
        r[s:s + chunk] = rr
        undefined[s:s + chunk] = undef
    return r, undefined


def summarize(truth, pred):
    out, per_gene = {}, {}
    for m in ("pearson", "spearman"):
        r, u = per_gene_correlation(truth, pred, m)
        out[f"mean_{m}"] = float(r.mean())
        out[f"n_undefined_{m}"] = int(u.sum())
        per_gene[m] = r
        per_gene[f"{m}_undefined"] = u
    out["n_genes_evaluated"] = int(truth.shape[1])
    out["n_cells_evaluated"] = int(truth.shape[0])
    return out, per_gene
