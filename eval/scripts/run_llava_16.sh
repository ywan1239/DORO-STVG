#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EVAL_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME="llava-1.6"
MODEL_PATH="/path/to/llava-v1.6-mistral-7b-hf"
DATA_NAME="dorostvg"
ANNOTATION_PATH="/path/to/query.jsonl"
VIDEO_DIR="/path/to/videos"
OUTPUT_DIR="./res/llava16_eval"
BATCH_SIZE=1
MAX_TOKENS=512
MAX_MODEL_LEN=8192
TEMPERATURE=0.0
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9

LLAVA_16_MAX_FRAMES=4
LLAVA_16_GRID_COLUMNS=2

echo "=========================================="
echo "LLaVA-1.6 STVG Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "CUDA Visible Devices:    $CUDA_VISIBLE_DEVICES"
echo "Batch Size:              $BATCH_SIZE"
echo "Max Tokens:              $MAX_TOKENS"
echo "Max Model Length:        $MAX_MODEL_LEN"
echo "Temperature:             $TEMPERATURE"
echo "Tensor Parallel Size:    $TENSOR_PARALLEL_SIZE"
echo "GPU Memory Utilization:  $GPU_MEMORY_UTILIZATION"
echo "Max Frames:              $LLAVA_16_MAX_FRAMES"
echo "Grid Columns:            $LLAVA_16_GRID_COLUMNS"
echo "=========================================="
echo ""

LLAVA_16_MAX_FRAMES="$LLAVA_16_MAX_FRAMES" \
LLAVA_16_GRID_COLUMNS="$LLAVA_16_GRID_COLUMNS" \
python main.py run \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --data_name "$DATA_NAME" \
  --annotation_path "$ANNOTATION_PATH" \
  --video_dir "$VIDEO_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_tokens "$MAX_TOKENS" \
  --max_model_len "$MAX_MODEL_LEN" \
  --temperature "$TEMPERATURE" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
