#!/bin/bash
# Generic SLURM helper (no project-specific paths): 30 combinations as an array, then merge.
# Usage: bash sbatch/run_selection.sh <train_raw_matrix> <test_dropout_matrix> <out_dir>
set -euo pipefail
TRAIN="$1"; TEST="$2"; OUT="$3"
ENV_SETUP="${SHIFTFORMER_ENV_SETUP:-source ~/.bashrc; conda activate vae_env2}"
mkdir -p "$OUT/parts" "$OUT/logs"
arr_id=$(sbatch --parsable --job-name=sf_combo --partition=gpu --gres=gpu:1 --time=1-00:00:00 \
  --mem=64G --cpus-per-task=1 --array=0-29 \
  --output="$OUT/logs/combo_%a.out" --error="$OUT/logs/combo_%a.err" \
  --wrap="$ENV_SETUP; shiftformer choose --train '$TRAIN' --test '$TEST' --combinations \$SLURM_ARRAY_TASK_ID --out_dir '$OUT/parts/combo_'\$SLURM_ARRAY_TASK_ID")
sbatch --parsable --dependency=afterany:$arr_id --job-name=sf_merge --partition=shared --time=0-01:00:00 \
  --mem=8G --output="$OUT/logs/merge.out" --error="$OUT/logs/merge.err" \
  --wrap="$ENV_SETUP; shiftformer merge --parts '$OUT'/parts/combo_* --out_dir '$OUT'"
