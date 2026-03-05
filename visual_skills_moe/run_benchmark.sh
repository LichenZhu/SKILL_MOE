#!/bin/bash
# run_benchmark.sh
# Usage: ./run_benchmark.sh [dataset_path] [model_name] [limit]

DATASET=${1:-"benchmarks/analysis/random50_sm_seed20260213.json"}
MODEL=${2:-"Qwen/Qwen2.5-Omni-7B"}
LIMIT=${3:-50}

# Fix RoPE tensor device mismatch by pinning to a single GPU with enough VRAM.
# Dynamically pick the GPU with the most free memory (needs >= 20 GiB).
BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t',' -k2 -rn | awk -F',' '{if ($2 > 20000) {print $1; exit}}')
if [ -z "$BEST_GPU" ]; then
    echo "ERROR: No GPU with >20 GiB free found. Aborting." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$BEST_GPU
export OPENAI_API_BASE="http://hl279-cmp-01.egr.duke.edu:4000/v1"

echo "Running benchmark on GPU $BEST_GPU (auto-selected)..."
echo "Dataset: $DATASET"
echo "Model: $MODEL"

python benchmark.py \
    --dataset "$DATASET" \
    --model_name "$MODEL" \
    --limit "$LIMIT"
