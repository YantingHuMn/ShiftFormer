#!/bin/bash
# Generic SLURM helper for the default workflow (rule-chosen normalization, log2(count+2)).
# Usage: bash sbatch/run_default.sh <raw_count_matrix> <out_dir> [reference_matrix]
# CPU with 1 thread reproduces the original One_Shifting Transformer_denoise runs.
set -euo pipefail
INPUT="$1"; OUT="$2"; REF="${3:-}"
ENV_SETUP="${SHIFTFORMER_ENV_SETUP:-source ~/.bashrc; conda activate vae_env2}"
mkdir -p "$OUT"
sbatch --parsable --job-name=shiftformer --partition=shared --time=2-00:00:00 --mem=64G --cpus-per-task=1 \
  --output="$OUT/shiftformer.log" \
  --wrap="$ENV_SETUP; shiftformer run --input '$INPUT' --out_dir '$OUT' ${REF:+--reference '$REF'} --threads 1 --device cpu"
