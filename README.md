# ShiftFormer (v1): transformation × normalization selection with Transformer denoise

This package checks whether the raw matrix and an externally supplied dropout matrix
from the same modality are enough to select a transformation–normalization combination.
It runs a fixed-hyperparameter Transformer denoiser on 6 transformations × 5
normalizations, with 5 outer folds for each of the 30 combinations.

The package is installed in editable mode in `vae_env2`. To install it elsewhere, run
`pip install --no-deps -e /dcs10/hongkai/data/yhu1/ShiftFormer`.

## Usage from One_Shifting

```python
import shiftformer
summary, report = shiftformer.select_trans_norm(
    raw="…/Count_Matrix_norm_by_no_norm.feather",      # raw matrix
    dropout="…/dropout_counts.feather",               # external dropout matrix
    out_dir="…/shiftformer_selection",
    pipeline_genes="…/Count_Matrix_norm_by_no_norm.feather")  # optional: gene set of the original pipeline
```

```bash
conda activate vae_env2
shiftformer select --raw RAW.feather --dropout DROP.feather --out_dir OUT          # all in one process
bash /dcs10/hongkai/data/yhu1/ShiftFormer/sbatch/run_selection.sh RAW DROP OUT [orientation] [pipeline_genes]  # SLURM
# Split form: shiftformer prepare / run --combo_index i / summarize
```

Combination index `i` selects `TRANS[i // 5]` with `NORM[i % 5]`:
- `TRANS = no_trans, count+1, sqrt, sqrt+1, log2 (=log2(x+1)), log2(count+2)`
- `NORM = no_norm, 1e6, 1e5, 1e4, 1e3`

## Input

Both inputs are count matrices with no normalization or transformation applied. The
dropout matrix is used as supplied; the package never applies dropout itself. Each
input can be:
- a `.feather` / `.parquet` / `.csv` / `.tsv` file, or
- a pandas DataFrame.

The first column holds the IDs and the header holds the column IDs. The default layout
is cells × genes; use `orientation="genes_x_cells"` for the other layout. The dropout
matrix is reordered to the raw matrix's cell and gene order.

The run stops with `StopForUser` (exit code 2 on the command line) and changes nothing
if any of these occur:
- duplicate or unmatched IDs
- NaN, Inf or negative values
- a cell whose library size is 0

The reason and the IDs involved are written to `run_log.json`.

## Output (only these three files are kept)

| File | Contents |
|---|---|
| `summary.tsv` | 30 rows: transformation, normalization, mean_pearson, mean_spearman, n_undefined_pearson, n_undefined_spearman |
| `report.txt` | The best combinations by Pearson and by Spearman, with all ties listed |
| `run_log.json` | Seed; the 5-fold split (outer-test cell IDs per fold, inner sizes, split rules); best epoch and epochs run for every fold of every combination; early-stopping settings; status of each combination; number of negative values after the inverse; number of all-zero reference genes; library-size gene set; saved precision; the rule for treating a vector as constant |

**How results are reported.**
- `report.txt` uses the label **FINAL best** only when all 30 combinations succeeded.
- If any combination failed or was not run:
  - that combination is `NA` in `summary.tsv`, and the report lists the reason;
  - the leaders among the successful combinations are labelled **PROVISIONAL leader**;
  - `_tmp` is kept, so `shiftformer run --combo_index <i>` re-runs only the failed combinations, and `shiftformer summarize` then rebuilds the outputs.
- Ties mean float64 scores that are exactly equal; every tied combination is listed.
- Scores are written with 17 significant digits (`%.17g` and Python `repr`), which round-trips exactly. In pandas, read the summary with `pd.read_csv(..., sep="\t", float_precision="round_trip")`.

**Library-size gene set.**
- The library size uses all genes in the input matrix. The raw and dropout gene sets must be identical; this is checked during alignment.
- `run_log.json` records the gene list, its SHA-256 hash, and the source file.
- When `pipeline_genes` is given, the input gene set is compared with the original pipeline's gene set and recorded as `identical` or `different` (with a WARNING and counts).
- Without `pipeline_genes`, the report says the gene set was NOT verified, so the library sizes should not be assumed to equal the original pipeline's.

Reconstruction matrices are written as temporary files under `OUT/_tmp` while metrics are
computed. They are deleted at the end unless you pass `keep_tmp=True`.

## Method (v1 conventions)

- **Library size:** each cell's sum over all genes of the aligned input matrix. This is the
  same formula as `R/normalize_and_save.R`, but the values equal the original pipeline's
  only if the gene sets are identical. The raw matrix uses raw counts and the
  dropout matrix uses dropout counts.
- **Preprocessing:** normalization is applied first, then transformation, each to the
  complete raw and dropout matrices. Rows are then taken by fold. This is valid because
  all 5 normalizations are computed per cell; any normalization that needs statistics
  across cells raises `NotImplementedError`.
- **Each fold:**
  - Training uses the processed raw matrix for the outer-train cells (80%).
  - Pass 1: early stopping on inner-validation MSE (patience 10, min_delta 0.001, at most
    100 epochs) picks the best epoch, and that model is discarded.
  - Pass 2: a fresh model is trained for that many epochs on all outer-train cells.
  - The processed dropout matrix for the outer-test cells (20%) is predicted.
  - The prediction is inverted: reverse transformation, then reverse normalization using
    those cells' dropout library size.
- **Evaluation:** after all folds, the predictions are concatenated by cell ID, and the
  package checks that each cell is predicted exactly once and that cell and gene order
  match the raw matrix. Pearson and Spearman correlations are computed per gene against
  the raw matrix. An undefined correlation (constant reference or prediction) counts as
  0, and the mean is taken over all genes. Metrics use the matrix saved as float32
  (float64 is optional).
- **Not included:** no gene or cell filtering, no clipping, and no hyperparameter search.
- **Model and training code:** copied unchanged from
  `One_Shifting/myproject/Step2_train_Transformer_denoise.py`.

Test: `python tests/test_synthetic.py --work_dir /scratch/dir`

## One_Shifting integration (`shiftformer one-shifting`)

Each call handles one sample, one input modality and one keep fraction. Paths come from the
One_Shifting config file, which is sourced in bash with `sample_name`, `folder_name` and
`dropout_keep_par` set.

```bash
module load conda_R; conda activate vae_env2
shiftformer one-shifting --config One_Shifting/inst/config/config_run_multi_ATAC_HCA_10x_ENCODE.sh \
    --folder_name HCA --sample_name HSB5871 --dropout_keep_par 0.9 [--check_only]
```

- **Raw input:** `${RAW_READ_DIR}/${V1}/Count_Matrix_norm_by_no_norm.feather`.
  - This is the existing Step1 output; it must be marked complete by `.count_matrix_done`.
  - The package never rebuilds it, and reads only this modality.
- **Dropout input:**
  - keep < 1: the bundled `R/dropout_multinomial.R` (a verbatim copy of One_Shifting's) is applied directly to the raw matrix with `set.seed(42)`. The genes and cells are not filtered again, so they match the raw matrix exactly. The result is written once to `<output>/input/`, first to a temporary file that is then renamed, with a `.dropout_done` marker.
  - keep = 1: the raw matrix itself is used.
- **Output:** `${READ_DIR}/Transformer_denoise_choose_combination/`, containing `summary.tsv`, `report.txt` and `run_log.json`.
  - A `.choose_combination_done` marker makes reruns skip.
  - A `.choose_combination_lock` lock stops two jobs running at once.
- **Standalone dropout:** `shiftformer make-dropout --raw RAW --out OUT --keep 0.9 [--seed 42]`.
