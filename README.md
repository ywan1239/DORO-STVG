# DORO-STVG

DORO-STVG provides an evaluation and data-generation toolkit for spatiotemporal video grounding. The evaluation side exposes a unified entry point, `eval/main.py`, for running STVG-style benchmarks across multiple video-language models with normalized temporal spans and frame-level bounding boxes.

This branch adds practical evaluation adapters for non-Qwen STVG backends while keeping Qwen as the baseline path:

- `qwen2.5vl` and `qwen3vl` use the baseline Qwen-VL route in `eval/models/qwen_family.py`.
- `videochat-r1` reuses the Qwen/vLLM inference stack but has its own clip sampling, prompt, and response normalization.
- `llava-1.6` is independent from VideoChat-R1 and shares only neutral STVG adapter utilities.
- `stvg-r1` uses the same Qwen-family runtime with STVG-R1-specific prompting and parsing.
- `llava-st-qwen2` uses the bundled LLaVA-ST runtime under `eval/dependences/LLaVAST`.
- `videomolmo` is supported through an external VideoMolmo runtime.

Shared frame sampling, `::split=start:end` parsing, JSON box parsing, and sampled-frame-to-original-frame remapping live in `eval/models/stvg_adapter_utils.py`.

## Evaluation Setup

Install `uv`, then create the model-specific evaluation environments from the checked-in lock files:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

cd /path/to/DORO-STVG
uv sync --project envs/eval/qwen
uv sync --project envs/eval/llavast
```

Use `envs/eval/qwen` for:

- `qwen2.5vl`
- `qwen3vl`
- `videochat-r1`
- `llava-1.6`
- `stvg-r1`
- `videomolmo`

Use `envs/eval/llavast` for:

- `llava-st-qwen2`

The virtual environments themselves are not committed. Recreate them with `uv sync`; commit only `pyproject.toml` and `uv.lock` when dependency definitions change.

## Running Evaluation

The unified command shape is:

```bash
cd /path/to/DORO-STVG
source envs/eval/qwen/.venv/bin/activate
bash eval/scripts/run_videochat_r1.sh
```

Edit the configuration block at the top of the script to set `MODEL_PATH`, `ANNOTATION_PATH`, `VIDEO_DIR`, `OUTPUT_DIR`, and `CUDA_VISIBLE_DEVICES`. The script follows the same launch style as `eval/scripts/run_qwen.sh`: it enters `eval/` and calls `python main.py run`.

`CUDA_VISIBLE_DEVICES` selects the GPU. `VIDEOCHAT_R1_MAX_FRAMES` and `VIDEOCHAT_R1_MAX_OUTPUT_FRAMES` are adapter controls for clip sampling and sparse frame outputs; keeping them explicit makes smoke-test results easier to reproduce.

For `llava-1.6`:

```bash
source envs/eval/qwen/.venv/bin/activate
bash eval/scripts/run_llava_16.sh
```

Edit the configuration block at the top of the script to set `MODEL_PATH`, `ANNOTATION_PATH`, `VIDEO_DIR`, `OUTPUT_DIR`, and `CUDA_VISIBLE_DEVICES`. The script follows the same launch style as `eval/scripts/run_qwen.sh`: it enters `eval/` and calls `python main.py run`.

`CUDA_VISIBLE_DEVICES` selects the GPU. `LLAVA_16_MAX_FRAMES` controls how many video frames are sampled into the image grid, and `LLAVA_16_GRID_COLUMNS` controls the grid layout. They are not required for import, but they should stay explicit in benchmark scripts because they change the model input.

For `stvg-r1`, benchmark runs should normally disable optional heuristic visual refinement:

```bash
CUDA_VISIBLE_DEVICES=0 \
STVG_R1_VISUAL_REFINEMENT=0 \
uv run --project envs/eval/qwen python eval/main.py run \
  --model_name stvg-r1 \
  --model_path /path/to/stvg-r1-model-7b \
  --data_name dorostvg \
  --annotation_path /path/to/query.jsonl \
  --video_dir /path/to/videos \
  --output_dir ./res/stvg_r1_eval \
  --batch_size 1 \
  --max_tokens 512 \
  --max_model_len 8192 \
  --temperature 0.0
```

For `llava-st-qwen2`:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run --project envs/eval/llavast python eval/main.py run \
  --model_name llava-st-qwen2 \
  --model_path /path/to/LLaVA-ST-Qwen2-7B \
  --data_name dorostvg \
  --annotation_path /path/to/query.jsonl \
  --video_dir /path/to/videos \
  --output_dir ./res/llava_st_qwen2_eval \
  --batch_size 1 \
  --max_tokens 512 \
  --max_model_len 8192 \
  --temperature 0.0
```

For `videomolmo`, point the adapter to an external VideoMolmo checkout:

```bash
export VIDEOMOLMO_REPO=/path/to/VideoMolmo
export VIDEOMOLMO_PYTHON=/path/to/videomolmo/bin/python
export VIDEOMOLMO_COMPACT_QUERY=1
```

Then run with `--model_name videomolmo --model_path videomolmo`.

## Outputs

Outputs are written under the selected `output_dir`. Each run creates:

- `results.jsonl`: per-sample raw response, parsed prediction, metadata, and metrics.
- `status.json`: run metadata, sample count, and averaged metrics.

## Data Engine

`graph_generator/` generates structured data from raw videos. The pipeline includes scene splitting, object detection and tracking, attribute generation, action detection, relation generation, optional cross-shot reference edges, STVG query generation, and conversion into training JSONL.

Relevant entry points:

- `graph_generator/main.py`: main scene graph generation entry.
- `graph_generator/modules/query_generator_cpsat.py`: query generation from scene graphs.
- `graph_generator/utils/format_train.py`: conversion into training format.
- `graph_generator/scripts/run_generator.sh`: command examples used by the data engine.

The graph generator uses separate `uv` environments:

```bash
cd /path/to/DORO-STVG
uv sync --project envs/graph_generator/main
uv sync --project envs/graph_generator/action_detector
```

It also requires external detector/action/model checkpoints and API credentials. See `graph_generator/README.md` for the full setup.

## Training Data Format

The training-friendly JSONL produced by `graph_generator/utils/format_train.py` contains fields such as:

- `videopath`
- `queryid`
- `query`
- `Difficulty`
- `Width` / `Height`
- `box`

The `box` field stores trajectories in this form:

```text
target description: <frame_idx, time_sec, x1, y1, x2, y2; ... />
```

Coordinates are normalized to `[0, 1]` using the video width and height.
