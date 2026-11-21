"""I/O, alignment, normalization, transformation and their inverses.

Forward order follows the original pipeline: raw -> normalization -> transformation.
  normalization  (R/normalize_and_save.R):  x_norm[c, g] = count[c, g] / lib_size[c] * S
  transformation (myproject/Step2_train_Transformer_denoise.py::_apply_trans)
Inverse order follows R/run_correlation_scatter_reverse.R (data_mode
v1_recon_reverse_trans_norm): reverse transformation FIRST, then reverse normalization.

All operations here are element-wise or per-cell (row-wise). None of them estimates a
parameter from other cells, so applying them to any subset of cells gives exactly the
same rows as applying them to the whole matrix.
"""
import os
import re

import numpy as np
import pandas as pd

TRANSFORMATIONS = ["no_trans", "count+1", "sqrt", "sqrt+1", "log2", "log2(count+2)"]
NORMALIZATIONS = ["no_norm", "1000000", "100000", "10000", "1000"]
# normalizations whose parameters are computed per cell (from that cell's own row only).
# Anything else (e.g. "standardize", which needs per-gene mean/SD across cells) is refused.
CELLWISE_NORMALIZATIONS = set(NORMALIZATIONS)
LIBRARY_SIZE_DEFINITION = ("per-cell sum of counts over ALL genes of the aligned input matrix; train (raw) "
                           "matrix -> raw counts, test (dropout) matrix -> dropout counts. Equal to an "
                           "external pipeline's library sizes only if that pipeline used the same gene "
                           "set (see library_size_gene_set.vs_original_pipeline)")


def sanitize(name):
    return re.sub(r"[^0-9A-Za-z._+-]", "_", str(name).strip())


def combo_tag(trans, norm):
    return f"trans_{sanitize(trans)}__norm_{sanitize(norm)}"


def all_combos():
    return [(t, n) for t in TRANSFORMATIONS for n in NORMALIZATIONS]


# ---------------------------------------------------------------- transformations
def apply_trans(m, trans):
    if trans == "no_trans":
        return m.copy()
    if trans == "count+1":
        return m + 1
    if trans == "sqrt":
        return np.sqrt(m)
    if trans == "sqrt+1":
        return np.sqrt(m + 1)
    if trans == "log2":
        return np.log2(m + 1)
    if trans == "log2(count+2)":
        return np.log2(m + 2)
    raise ValueError(f"Unknown transformation: {trans}")


def reverse_trans(m, trans):
    # identical to reverse_trans() in R/run_correlation_scatter_reverse.R
    if trans == "no_trans":
        return m.copy()
    if trans == "count+1":
        return m - 1
    if trans == "sqrt":
        return m ** 2
    if trans == "sqrt+1":
        return m ** 2 - 1
    if trans == "log2":
        return 2.0 ** m - 1
    if trans == "log2(count+2)":
        return 2.0 ** m - 2
    raise ValueError(f"Unknown transformation: {trans}")


# ---------------------------------------------------------------- normalizations
def library_size(m):
    """Per-cell library size = row sum over all genes of the matrix given."""
    return m.sum(axis=1)


def apply_norm(m, norm, lib_size):
    if norm not in CELLWISE_NORMALIZATIONS:
        raise NotImplementedError(
            f"normalization '{norm}' is not a per-cell library-size normalization; it would need "
            "parameters estimated across cells. Not implemented by design - decide how to estimate them first.")
    if norm == "no_norm":
        return m.copy()
    S = float(norm)
    if np.any(lib_size <= 0):
        raise ValueError("library size <= 0 for some cell; cannot normalize")
    return m / lib_size[:, None] * S


def reverse_norm(m, norm, lib_size):
    if norm == "no_norm":
        return m.copy()
    S = float(norm)
    return m / S * lib_size[:, None]


def process_full(raw, drop, trans, norm):
    """Apply one combination to the COMPLETE raw and dropout matrices (norm, then trans).
    Each matrix uses its own per-cell library size. Valid before splitting because every
    allowed normalization is per cell."""
    lib_raw, lib_drop = library_size(raw), library_size(drop)
    z_raw = apply_trans(apply_norm(raw, norm, lib_raw), trans)
    z_drop = apply_trans(apply_norm(drop, norm, lib_drop), trans)
    return z_raw, z_drop, lib_raw, lib_drop


def forward(m, norm, trans):
    """raw counts (cells x genes) -> model space. Uses each cell's own library size."""
    lib = library_size(m)
    return apply_trans(apply_norm(m, norm, lib), trans), lib


def inverse(z, norm, trans, lib_size):
    """model space -> count scale. lib_size must be the one used in forward()."""
    return reverse_norm(reverse_trans(z, trans), norm, lib_size)


# ---------------------------------------------------------------- I/O
def read_matrix(path, orientation="cells_x_genes"):
    """Return (cell_ids, gene_ids, values float64, id_column_name).

    `path` may be
      - an AnnData (obs_names = cells, var_names = genes; X dense or sparse; always cells x genes),
      - a pandas DataFrame: if its first column is non-numeric it holds the row IDs (e.g. "pos"),
        otherwise the index holds them; column labels are the column IDs,
      - a file (.feather, .parquet, .csv, .tsv/.txt, optionally .gz): first column = row IDs,
        header = column IDs.
    """
    if hasattr(path, "obs_names") and hasattr(path, "var_names") and hasattr(path, "X"):
        X = path.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        return (np.array(path.obs_names, dtype=str), np.array(path.var_names, dtype=str),
                np.asarray(X, dtype=np.float64), "cell_id")
    if isinstance(path, pd.DataFrame):
        p = "<DataFrame>"
        if path.shape[1] > 0 and not pd.api.types.is_numeric_dtype(path.iloc[:, 0]):
            df = path
        else:
            df = path.reset_index()
            df.columns = [path.index.name or "cell_id"] + list(path.columns)
        header = [str(c) for c in df.columns]
    else:
        p = str(path)
    if isinstance(path, pd.DataFrame):
        pass
    elif p.endswith(".feather"):
        df = pd.read_feather(p)
        header = [str(c) for c in df.columns]
    elif p.endswith(".parquet"):
        df = pd.read_parquet(p)
        header = [str(c) for c in df.columns]
    elif re.search(r"\.(csv|tsv|txt)(\.gz)?$", p):
        import csv
        import gzip
        sep = "," if re.search(r"\.csv(\.gz)?$", p) else "\t"
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", newline="") as fh:
            header = next(csv.reader(fh, delimiter=sep))   # verbatim: pandas would rename duplicates
        df = pd.read_csv(p, sep=sep, header=None, skiprows=1, keep_default_na=False,
                         dtype={0: str}, na_values=[""])
        if df.shape[1] != len(header):
            raise ValueError(f"{p}: header has {len(header)} fields but rows have {df.shape[1]}")
    else:
        raise ValueError(f"Unsupported file extension: {p}")
    id_col = header[0]
    row_ids = df.iloc[:, 0].astype(str).to_numpy()
    col_ids = np.array(header[1:], dtype=str)
    body = df.iloc[:, 1:]
    non_numeric = [c for c in body.columns if not pd.api.types.is_numeric_dtype(body[c])]
    if non_numeric:
        raise ValueError(f"{p}: non-numeric columns {non_numeric[:5]}")
    values = body.to_numpy(dtype=np.float64)
    if orientation == "cells_x_genes":
        return row_ids, col_ids, values, str(id_col)
    if orientation == "genes_x_cells":
        return col_ids, row_ids, values.T.copy(), "cell_id"
    raise ValueError(f"orientation must be cells_x_genes or genes_x_cells, got {orientation}")


def read_gene_ids(source, orientation="cells_x_genes"):
    """Gene IDs from: a list/array, a .txt file (one ID per line, no header), or a
    matrix file in the same layout as the inputs (only the gene IDs are read)."""
    if not isinstance(source, (str, os.PathLike)):
        return np.array([str(g) for g in source], dtype=str)
    p = str(source)
    if re.search(r"\.txt(\.gz)?$", p) and not _looks_like_matrix(p):
        return pd.read_csv(p, header=None, dtype=str, keep_default_na=False)[0].to_numpy(dtype=str)
    if orientation == "cells_x_genes":
        if p.endswith(".feather"):
            import pyarrow.ipc as ipc
            names = ipc.open_file(p).schema.names
        elif p.endswith(".parquet"):
            import pyarrow.parquet as pq
            names = pq.read_schema(p).names
        else:
            import csv
            import gzip
            with (gzip.open if p.endswith(".gz") else open)(p, "rt", newline="") as fh:
                names = next(csv.reader(fh, delimiter="," if re.search(r"\.csv(\.gz)?$", p) else "\t"))
        return np.array([str(n) for n in names[1:]], dtype=str)
    _, genes, _, _ = read_matrix(p, orientation)
    return genes


def _looks_like_matrix(p):
    import gzip
    with (gzip.open if p.endswith(".gz") else open)(p, "rt") as fh:
        first = fh.readline()
    return "\t" in first


def write_matrix(path, cell_ids, gene_ids, values, id_col):
    df = pd.DataFrame(values, columns=list(gene_ids))
    df.insert(0, id_col, list(cell_ids))
    df.to_feather(path)


def _dups(ids):
    s = pd.Series(ids)
    return sorted(s[s.duplicated(keep=False)].unique().tolist())


def check_alignment(raw_cells, raw_genes, drop_cells, drop_genes):
    """Compare ID sets. Never drops anything: returns a report and (if alignable)
    the permutation that puts the dropout matrix in raw-matrix order."""
    rep = {
        "raw_n_cells": int(len(raw_cells)), "raw_n_genes": int(len(raw_genes)),
        "dropout_n_cells": int(len(drop_cells)), "dropout_n_genes": int(len(drop_genes)),
        "raw_duplicate_cells": _dups(raw_cells), "raw_duplicate_genes": _dups(raw_genes),
        "dropout_duplicate_cells": _dups(drop_cells), "dropout_duplicate_genes": _dups(drop_genes),
    }
    rc, rg, dc, dg = set(raw_cells), set(raw_genes), set(drop_cells), set(drop_genes)
    rep["cells_only_in_raw"] = sorted(rc - dc)
    rep["cells_only_in_dropout"] = sorted(dc - rc)
    rep["genes_only_in_raw"] = sorted(rg - dg)
    rep["genes_only_in_dropout"] = sorted(dg - rg)
    problems = [k for k in ("raw_duplicate_cells", "raw_duplicate_genes", "dropout_duplicate_cells",
                            "dropout_duplicate_genes", "cells_only_in_raw", "cells_only_in_dropout",
                            "genes_only_in_raw", "genes_only_in_dropout") if rep[k]]
    rep["problems"] = problems
    rep["ok"] = not problems
    if rep["ok"]:
        cpos = {c: i for i, c in enumerate(drop_cells)}
        gpos = {g: i for i, g in enumerate(drop_genes)}
        rep["dropout_cell_order_matches_raw"] = bool(np.array_equal(raw_cells, drop_cells))
        rep["dropout_gene_order_matches_raw"] = bool(np.array_equal(raw_genes, drop_genes))
        rep["_cell_perm"] = np.array([cpos[c] for c in raw_cells])
        rep["_gene_perm"] = np.array([gpos[g] for g in raw_genes])
    return rep
