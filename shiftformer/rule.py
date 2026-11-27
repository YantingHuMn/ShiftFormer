"""Input-only rule that picks the normalization N for Transformer denoise with the
transformation fixed to log2(count + 2).

  Pearson : log10 N = 2.048 + 0.781 * log10(median number of detected genes per cell)
            then snapped to the nearest option on the log10 scale:
              1e3 -> 3, 1e4 -> 4, 1e5 -> 5, 1e6 -> 6, no_norm -> log10(median library size)
            (ties -> the option listed first in NORM_OPTIONS)
  Spearman: fixed N = 1e6

detected genes of a cell = number of genes with count > 0 in that cell
library size of a cell   = sum of its counts over all genes of the input matrix
Both medians are taken over the cells of the raw input count matrix (no reference needed).

The coefficients were fitted once, on 64 real samples (32 biosamples x RNA/ATAC; HCA, ENCODE,
10x), as the line that puts the most samples inside the set of norms whose Pearson score is
within 2% of the plot range of the best norm (leave-one-biosample-out: 50/64 acceptable).
They are constants of the package: nothing is refitted on the user's data.
"""
import numpy as np

TRANSFORMATION = "log2(count+2)"
PEARSON_INTERCEPT = 2.048
PEARSON_SLOPE = 0.781
SPEARMAN_NORM = "1000000"
NORM_OPTIONS = ["1000", "10000", "100000", "1000000", "no_norm"]   # order = tie-break order
RULE_TEXT = {
    "pearson": f"log10 N = {PEARSON_INTERCEPT} + {PEARSON_SLOPE} * log10(median detected genes per cell), "
               "snapped to the nearest of 1e3/1e4/1e5/1e6/no_norm (no_norm at log10(median library size))",
    "spearman": f"fixed N = {SPEARMAN_NORM}",
}


def input_features(counts):
    """counts: cells x genes raw count matrix (numpy). Returns the statistics the rule uses."""
    counts = np.asarray(counts)
    det = (counts > 0).sum(axis=1)
    lib = counts.sum(axis=1, dtype=np.float64)
    return {"n_cells": int(counts.shape[0]), "n_genes": int(counts.shape[1]),
            "median_detected_genes": float(np.median(det)),
            "median_library_size": float(np.median(lib))}


def choose_norm(counts=None, features=None):
    """Return {"pearson": {...}, "spearman": {...}, "features": {...}}.
    Give either the raw count matrix (cells x genes) or precomputed `features`."""
    f = dict(features) if features is not None else input_features(counts)
    if f["median_detected_genes"] <= 0 or f["median_library_size"] <= 0:
        raise ValueError("median detected genes / median library size must be > 0 to apply the rule")
    pred = PEARSON_INTERCEPT + PEARSON_SLOPE * np.log10(f["median_detected_genes"])
    pos = np.array([3.0, 4.0, 5.0, 6.0, np.log10(f["median_library_size"])])
    k = int(np.argmin(np.abs(pos - pred)))
    return {
        "transformation": TRANSFORMATION,
        "features": f,
        "pearson": {"normalization": NORM_OPTIONS[k], "predicted_log10N": float(pred),
                    "option_positions_log10": dict(zip(NORM_OPTIONS, map(float, pos))),
                    "rule": RULE_TEXT["pearson"]},
        "spearman": {"normalization": SPEARMAN_NORM, "rule": RULE_TEXT["spearman"]},
    }
