import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from .stvg_adapter_utils import (
    _extract_video_spec,
    _extract_json_candidate,
    _load_sampled_frames,
    _remap_frame_map,
)


logger = logging.getLogger(__name__)


def _build_frame_grid(frames, out_path: Path, columns: int) -> None:
    num_frames = len(frames)
    if num_frames <= 0:
        raise ValueError("No frames to build LLaVA 1.6 grid.")

    frame_height, frame_width = frames[0].shape[0], frames[0].shape[1]
    columns = max(1, min(columns, num_frames))
    rows = int(math.ceil(num_frames / columns))

    canvas = Image.new("RGB", (columns * frame_width, rows * frame_height), color=(255, 255, 255))
    drawer = ImageDraw.Draw(canvas)

    for idx, frame in enumerate(frames):
        row = idx // columns
        col = idx % columns
        x0 = col * frame_width
        y0 = row * frame_height
        frame_image = Image.fromarray(frame).convert("RGB")
        canvas.paste(frame_image, (x0, y0))

        label = f"{idx}"
        drawer.rectangle([x0 + 6, y0 + 6, x0 + 42, y0 + 30], fill=(0, 0, 0))
        drawer.text((x0 + 14, y0 + 10), label, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def _coerce_single_box(value) -> List[float] | None:
    if not isinstance(value, list):
        return None

    # Shape: [x1, y1, x2, y2]
    if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        return [float(v) for v in value]

    # Shape: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
    candidate_boxes = []
    for item in value:
        if isinstance(item, list) and len(item) == 4 and all(isinstance(v, (int, float)) for v in item):
            candidate_boxes.append([float(v) for v in item])

    if candidate_boxes:
        # Current LLaVA output often duplicates the same box twice.
        return candidate_boxes[0]

    return None


def _parse_llava16_frame_map(raw_text: str) -> Dict[int, List[float]]:
    json_candidate = _extract_json_candidate(raw_text)
    if not json_candidate:
        return {}

    try:
        payload = json.loads(json_candidate)
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    parsed: Dict[int, List[float]] = {}

    # Expected main shape:
    # {
    #   "target": {
    #     "0": [[...], [...]],
    #     "1": [[...], [...]]
    #   }
    # }
    target_payload = payload.get("target")
    if isinstance(target_payload, dict):
        for tile_idx_text, coords in target_payload.items():
            try:
                tile_idx = int(str(tile_idx_text).strip())
            except ValueError:
                continue
            box = _coerce_single_box(coords)
            if box is not None:
                parsed[tile_idx] = box
        if parsed:
            return parsed

    # Fallback nested shape:
    # {
    #   "0": {"target": {...}},
    #   "1": {"target": {...}}
    # }
    for tile_idx_text, tile_payload in payload.items():
        try:
            tile_idx = int(str(tile_idx_text).strip())
        except ValueError:
            continue

        if not isinstance(tile_payload, dict):
            continue

        nested_target = tile_payload.get("target")
        if not isinstance(nested_target, dict):
            continue

        for _, coords in nested_target.items():
            box = _coerce_single_box(coords)
            if box is not None:
                parsed[tile_idx] = box
                break

    return parsed


class Llava16Model:
    prompt_style = "llava_16"

    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        max_tokens: int = 512,
        max_model_len: int = 8192,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        del max_model_len, tensor_parallel_size, gpu_memory_utilization

        self.model_path = model_path
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_frames = int(os.getenv("LLAVA_16_MAX_FRAMES", "6"))
        self.grid_columns = int(os.getenv("LLAVA_16_GRID_COLUMNS", "3"))
        self.keep_tmp = os.getenv("LLAVA_16_KEEP_TMP", "0").lower() in {"1", "true", "yes"}

        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []
        self.last_video_frame_indices: List[List[int]] = []

        self.model = None
        self.processor = None
        self.torch = None
        self.device = None
        self.load_model()

    def load_model(self):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        model = None
        last_exc: Exception | None = None

        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
        except Exception as exc:
            last_exc = exc

        if model is None:
            for class_name in ("LlavaNextForConditionalGeneration", "LlavaForConditionalGeneration"):
                try:
                    cls = getattr(__import__("transformers", fromlist=[class_name]), class_name)
                except Exception:
                    continue
                try:
                    model = cls.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True,
                    )
                    break
                except Exception as exc:
                    last_exc = exc

        if model is None:
            raise RuntimeError(
                f"Failed to load LLaVA 1.6 model from {self.model_path}. Last exception: {last_exc}"
            )

        model.eval()
        self.model = model
        self.device = next(model.parameters()).device

        logger.info(
            "Initialized llava-1.6 adapter | model=%s device=%s max_frames=%s grid_columns=%s",
            self.model_path,
            self.device,
            self.max_frames,
            self.grid_columns,
        )

    def _augment_prompt(self, query: str, num_frames: int) -> str:
        allowed = ", ".join(str(i) for i in range(num_frames))
        suffix = (
            "\n\nImage layout notes:\n"
            "- The image is a collage of sampled frames from one short video clip.\n"
            "- Each tile has a numeric id drawn in its top-left corner.\n"
            "- Return exactly one JSON object with one top-level key named \"target\".\n"
            "- The value of target must map tile ids directly to bounding boxes.\n"
            "- Use at most one box per tile.\n"
            "- Do not return nested objects under a tile id.\n"
            "- Use only allowed tile ids as keys.\n"
            "- Do not output any tile id outside the allowed set.\n"
            "- If the target is absent in a tile, omit that tile.\n"
            "- Each bounding box must be normalized to [0, 1] within that tile only, not across the whole collage.\n"
            f"- Allowed tile ids: [{allowed}]\n"
        )
        return query.rstrip() + suffix

    def _generate_one(self, prompt: str, image: Image.Image, system_prompt: str) -> str:
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        try:
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = f"{system_prompt}\n<image>\n{prompt}"

        inputs = self.processor(images=image, text=text, return_tensors="pt")
        moved_inputs: Dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved_inputs[key] = value.to(self.device)
            else:
                moved_inputs[key] = value

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        with self.torch.inference_mode():
            output_ids = self.model.generate(**moved_inputs, **generate_kwargs)

        input_len = moved_inputs["input_ids"].shape[1] if "input_ids" in moved_inputs else 0
        generated_ids = output_ids[:, input_len:] if input_len and output_ids.shape[1] > input_len else output_ids
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""

    def _run_one(self, query: str, video_input: str, system_prompt: str) -> Tuple[str, str, List[int]]:
        video_path, split = _extract_video_spec(video_input)
        frames, sampled_indices, width, height = _load_sampled_frames(
            video_path,
            split=split,
            max_frames=self.max_frames,
        )

        tmp_root = Path(tempfile.mkdtemp(prefix="llava16_eval_"))
        grid_path = tmp_root / "frame_grid.jpg"
        try:
            _build_frame_grid(frames, grid_path, columns=self.grid_columns)
            image = Image.open(grid_path).convert("RGB")
            raw_text = self._generate_one(
                prompt=self._augment_prompt(query, num_frames=len(sampled_indices)),
                image=image,
                system_prompt=system_prompt,
            )

            parsed_frame_map = _parse_llava16_frame_map(raw_text)
            logger.info("LLaVA 1.6 parsed %d frame boxes from raw response", len(parsed_frame_map))

            filtered_frame_map = {
                frame_idx: coords
                for frame_idx, coords in parsed_frame_map.items()
                if 0 <= frame_idx < len(sampled_indices)
            }

            remapped = _remap_frame_map(
                filtered_frame_map,
                original_indices=sampled_indices,
                width=width,
                height=height,
            )

            normalized_output = json.dumps({"target": remapped}, ensure_ascii=False)
            return normalized_output, raw_text, sampled_indices
        finally:
            if not self.keep_tmp:
                shutil.rmtree(tmp_root, ignore_errors=True)

    def predict_batch(
        self,
        queries: List[str],
        video_paths: List[str],
        system_prompt: str,
    ) -> List[str]:
        self.last_user_prompts = list(queries)

        normalized_outputs: List[str] = []
        raw_outputs: List[str] = []
        frame_indices: List[List[int]] = []

        for query, video_path in zip(queries, video_paths):
            normalized_output, raw_output, sampled_indices = self._run_one(
                query=query,
                video_input=video_path,
                system_prompt=system_prompt,
            )
            normalized_outputs.append(normalized_output)
            raw_outputs.append(raw_output)
            frame_indices.append(sampled_indices)

        self.last_raw_responses = raw_outputs
        self.last_video_frame_indices = frame_indices
        return normalized_outputs
