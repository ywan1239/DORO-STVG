#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-3}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_smoke1.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/sdc/xingjianwang/yibowang/DORO-STVG/res/smoke_suite_gpu${GPU_ID}}"

QWEN_ENV="${QWEN_ENV:-envs/eval/qwen}"
LLAVAST_ENV="${LLAVAST_ENV:-envs/eval/llavast}"
LLAVAST_SOURCE_DIR="${LLAVAST_SOURCE_DIR:-/mnt/sdc/xingjianwang/yibowang/LLaVA-ST}"

QWEN25_MODEL_PATH="${QWEN25_MODEL_PATH:-}"
QWEN3_MODEL_PATH="${QWEN3_MODEL_PATH:-}"
LLAVAST_MODEL_PATH="${LLAVAST_MODEL_PATH:-}"
VIDEOCHAT_R1_MODEL_PATH="${VIDEOCHAT_R1_MODEL_PATH:-}"
LLAVA16_MODEL_PATH="${LLAVA16_MODEL_PATH:-}"
STVG_R1_MODEL_PATH="${STVG_R1_MODEL_PATH:-}"

run_eval() {
  local model_name="$1"
  local model_path="$2"
  local output_dir="$3"
  shift 3

  echo "============================================================"
  echo "Running ${model_name}"
  echo "Model path: ${model_path}"
  echo "Output dir: ${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  uv run --project "${QWEN_ENV}" python eval/main.py run \
    --model_name "${model_name}" \
    --model_path "${model_path}" \
    --data_name dorostvg \
    --annotation_path "${ANNOTATION_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --output_dir "${output_dir}" \
    --batch_size 1 \
    --max_tokens 512 \
    --max_model_len 8192 \
    --temperature 0.0 \
    "$@"
}

mkdir -p "${OUTPUT_BASE}"

if [[ -n "${QWEN25_MODEL_PATH}" ]]; then
  run_eval "qwen2.5vl" "${QWEN25_MODEL_PATH}" "${OUTPUT_BASE}/qwen25"
fi

if [[ -n "${QWEN3_MODEL_PATH}" ]]; then
  run_eval "qwen3vl" "${QWEN3_MODEL_PATH}" "${OUTPUT_BASE}/qwen3"
fi

if [[ -n "${VIDEOCHAT_R1_MODEL_PATH}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  VIDEOCHAT_R1_MAX_FRAMES="${VIDEOCHAT_R1_MAX_FRAMES:-32}" \
  VIDEOCHAT_R1_MAX_OUTPUT_FRAMES="${VIDEOCHAT_R1_MAX_OUTPUT_FRAMES:-8}" \
  uv run --project "${QWEN_ENV}" python eval/main.py run \
    --model_name videochat-r1 \
    --model_path "${VIDEOCHAT_R1_MODEL_PATH}" \
    --data_name dorostvg \
    --annotation_path "${ANNOTATION_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --output_dir "${OUTPUT_BASE}/videochat_r1" \
    --batch_size 1 \
    --max_tokens 512 \
    --max_model_len 8192 \
    --temperature 0.0
fi

if [[ -n "${LLAVA16_MODEL_PATH}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  LLAVA_16_MAX_FRAMES="${LLAVA_16_MAX_FRAMES:-4}" \
  LLAVA_16_GRID_COLUMNS="${LLAVA_16_GRID_COLUMNS:-2}" \
  uv run --project "${QWEN_ENV}" python eval/main.py run \
    --model_name llava-1.6 \
    --model_path "${LLAVA16_MODEL_PATH}" \
    --data_name dorostvg \
    --annotation_path "${ANNOTATION_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --output_dir "${OUTPUT_BASE}/llava16" \
    --batch_size 1 \
    --max_tokens 512 \
    --max_model_len 8192 \
    --temperature 0.0
fi

if [[ -n "${STVG_R1_MODEL_PATH}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  STVG_R1_MAX_FRAMES="${STVG_R1_MAX_FRAMES:-32}" \
  STVG_R1_MAX_OUTPUT_FRAMES="${STVG_R1_MAX_OUTPUT_FRAMES:-8}" \
  STVG_R1_VISUAL_REFINEMENT="${STVG_R1_VISUAL_REFINEMENT:-1}" \
  uv run --project "${QWEN_ENV}" python eval/main.py run \
    --model_name stvg-r1 \
    --model_path "${STVG_R1_MODEL_PATH}" \
    --data_name dorostvg \
    --annotation_path "${ANNOTATION_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --output_dir "${OUTPUT_BASE}/stvg_r1" \
    --batch_size 1 \
    --max_tokens 512 \
    --max_model_len 8192 \
    --temperature 0.0
fi

if [[ -n "${LLAVAST_MODEL_PATH}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  LLAVA_ST_SOURCE_DIR="${LLAVAST_SOURCE_DIR}" \
  uv run --project "${LLAVAST_ENV}" python eval/main.py run \
    --model_name llava-st-qwen2 \
    --model_path "${LLAVAST_MODEL_PATH}" \
    --data_name dorostvg \
    --annotation_path "${ANNOTATION_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --output_dir "${OUTPUT_BASE}/llava_st_qwen2" \
    --batch_size 1 \
    --max_tokens 512 \
    --max_model_len 8192 \
    --temperature 0.0
fi

if [[ -n "${VIDEOMOLMO_REPO:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  VIDEOMOLMO_REPO="${VIDEOMOLMO_REPO}" \
  VIDEOMOLMO_PYTHON="${VIDEOMOLMO_PYTHON:-python}" \
  VIDEOMOLMO_COMPACT_QUERY="${VIDEOMOLMO_COMPACT_QUERY:-1}" \
  uv run --project "${QWEN_ENV}" python eval/main.py run \
    --model_name videomolmo \
    --model_path videomolmo \
    --data_name dorostvg \
    --annotation_path "${ANNOTATION_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --output_dir "${OUTPUT_BASE}/videomolmo" \
    --batch_size 1 \
    --max_tokens 512 \
    --max_model_len 8192 \
    --temperature 0.0
fi

echo "Smoke suite finished. Outputs under: ${OUTPUT_BASE}"
