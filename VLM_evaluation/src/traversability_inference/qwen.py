"""Lazy, single-device Qwen adapters for raw current-frame classification."""

from __future__ import annotations

import gc
import hashlib
import importlib
import re
import threading
from collections import defaultdict

from .configuration import JSON_PROMPT, MODEL_SPECS, normalize_settings
from .preparation import prepare_target, resize_aligned
from .records import InferenceError, parse_response, prediction


_EXCLUDED_MODULES = ("model.visual", "lm_head")
_MODEL_CLASSES = {
    "qwen3_vl_4b": "Qwen3VLForConditionalGeneration",
    "qwen3_5_4b": "Qwen3_5ForConditionalGeneration",
    "qwen3_5_2b": "Qwen3_5ForConditionalGeneration",
}


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _portable_processor_config(value):
    if isinstance(value, dict):
        return {
            key: _portable_processor_config(item)
            for key, item in value.items()
            if key not in {"cache_dir", "_name_or_path", "name_or_path", "pretrained_model_name_or_path"}
        }
    if isinstance(value, list):
        return [_portable_processor_config(item) for item in value]
    return value


def _tolist(value):
    return value.tolist() if hasattr(value, "tolist") else value


class QwenBackend:
    """Load the explicitly selected Qwen checkpoint; no automatic fallback."""

    def __init__(self, model_key: str, settings: dict):
        if model_key not in _MODEL_CLASSES:
            raise ValueError(f"Unsupported Qwen model key: {model_key}")
        self.model_key = model_key
        self.settings = normalize_settings(model_key, settings)
        self.model = None
        self.processor = None
        self._closed = False
        self._lock = threading.RLock()
        self._torch = None
        self.metadata = {}
        try:
            self._load()
        except Exception:
            self.close()
            raise

    def _load(self):
        # These imports can initialize CUDA and are intentionally confined to loading.
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        bnb = importlib.import_module("bitsandbytes")
        self._torch = torch
        requested_device = self.settings.get("device", "cuda:0")
        if not isinstance(requested_device, str) or not re.fullmatch(r"cuda:\d+", requested_device):
            raise ValueError("Qwen requires one explicit CUDA device, such as cuda:0.")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen 4-bit inference requires an available CUDA GPU.")
        self._device = torch.device(requested_device)
        index = int(requested_device.split(":")[1])
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {requested_device}")
        spec = MODEL_SPECS[self.model_key]
        common = {
            "revision": spec["revision"],
            "local_files_only": self.settings.get("local_files_only", False),
        }
        if self.settings.get("cache_dir") is not None:
            common["cache_dir"] = self.settings["cache_dir"]
        with torch.cuda.device(self._device):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quantization = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
                llm_int8_skip_modules=list(_EXCLUDED_MODULES),
                llm_int8_enable_fp32_cpu_offload=False,
            )
            self.processor = transformers.AutoProcessor.from_pretrained(spec["checkpoint"], **common)
            model_class = getattr(transformers, _MODEL_CLASSES[self.model_key])
            self.model = model_class.from_pretrained(
                spec["checkpoint"],
                **common,
                dtype=dtype,
                device_map={"": requested_device},
                quantization_config=quantization,
                attn_implementation="sdpa",
            )
            self.model.eval()
            audit = self._audit_loaded_model(torch, bnb)
        image_processor = self.processor.image_processor
        self._patch_size = int(image_processor.patch_size)
        self._merge_size = int(image_processor.merge_size)
        if self._patch_size <= 0 or self._merge_size <= 0:
            raise RuntimeError("Invalid Qwen image processor alignment settings.")
        self._factor = self._patch_size * self._merge_size
        image_config = image_processor.to_dict()
        chat_template = getattr(self.processor, "chat_template", None)
        if chat_template is None:
            chat_template = getattr(self.processor.tokenizer, "chat_template", "")
        self.metadata = {
            "adapter": "qwen",
            "checkpoint": spec["checkpoint"],
            "revision": spec["revision"],
            "model_class": type(self.model).__name__,
            "device": requested_device,
            "attention_implementation": "sdpa",
            "quantization": {
                "language_bits": 4,
                "type": "nf4",
                "double_quantization": True,
                "compute_dtype": str(dtype),
                "excluded_modules": list(_EXCLUDED_MODULES),
                "cpu_offload": False,
            },
            "processor": {
                "class": type(self.processor).__name__,
                "image_processor_class": type(image_processor).__name__,
                "tokenizer_class": type(self.processor.tokenizer).__name__,
                "image_processor_config": _portable_processor_config(_jsonable(image_config)),
                "chat_template_sha256": hashlib.sha256(str(chat_template).encode("utf-8")).hexdigest(),
                "patch_size": self._patch_size,
                "merge_size": self._merge_size,
                "alignment_factor": self._factor,
                "do_resize": False,
                "scene_max_pixels": self.settings["scene_visual_token_budget"] * self._factor**2,
                "crop_max_pixels": self.settings["crop_visual_token_budget"] * self._factor**2,
                "token_verification": "grid_product_div_merge_squared_and_image_placeholders",
            },
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "repetition_penalty": 1.0,
                "max_new_tokens": self.settings["max_new_tokens"],
                "use_cache": True,
                "request_concurrency": 1,
                "seed": self.settings["seed"],
                "enable_thinking": False,
            },
            "audit": audit,
        }

    def _audit_loaded_model(self, torch, bnb):
        quantized = []
        floating_language_linears = []
        vision_modules = []
        for name, module in self.model.named_modules():
            is_quantized = isinstance(module, bnb.nn.Linear4bit)
            if is_quantized:
                quantized.append(name)
            if name == "model.visual" or name.startswith("model.visual."):
                vision_modules.append(name)
                if is_quantized:
                    raise RuntimeError(f"Vision module was quantized despite exclusion: {name}")
            if name == "lm_head" and is_quantized:
                raise RuntimeError("The excluded lm_head was quantized.")
            if name.startswith("model.language_model.") and isinstance(module, torch.nn.Linear) and not is_quantized:
                floating_language_linears.append(name)
        language_quantized = [name for name in quantized if name.startswith("model.language_model.")]
        if not language_quantized:
            raise RuntimeError("No quantized language linear modules were found.")
        if floating_language_linears:
            raise RuntimeError(f"Language linear modules remain unquantized: {floating_language_linears}")
        if not vision_modules:
            raise RuntimeError("Expected floating-point model.visual was not found.")
        placements, dtypes, vision_dtypes = defaultdict(list), defaultdict(list), defaultdict(list)
        vision_parameter_count = 0
        for name, parameter in self.model.named_parameters():
            placement = str(parameter.device)
            if placement != str(self._device):
                raise RuntimeError(f"Parameter {name} is on {placement}; expected {self._device} without offload.")
            placements[placement].append(name)
            dtypes[str(parameter.dtype)].append(name)
            if name.startswith("model.visual."):
                vision_parameter_count += 1
                if not parameter.is_floating_point():
                    raise RuntimeError(f"Vision parameter is not floating point: {name}")
                vision_dtypes[str(parameter.dtype)].append(name)
            if name.startswith("lm_head.") and not parameter.is_floating_point():
                raise RuntimeError(f"The excluded output head is not floating point: {name}")
        if not vision_parameter_count:
            raise RuntimeError("Expected vision parameters were not found.")
        for name, device in getattr(self.model, "hf_device_map", {}).items():
            normalized = f"cuda:{device}" if isinstance(device, int) else str(device)
            if normalized != str(self._device):
                raise RuntimeError(f"Module {name} is dispatched to {device}; automatic offload is prohibited.")
        return {
            "parameter_placement": dict(placements),
            "parameter_dtypes": dict(dtypes),
            "vision_parameter_dtypes": dict(vision_dtypes),
            "quantized_modules": quantized,
            "language_quantized_modules": language_quantized,
            "floating_language_linear_modules": floating_language_linears,
            "vision_is_floating_point": True,
        }

    def _prepare_inputs(self, target):
        scene_budget = self.settings["scene_visual_token_budget"]
        crop_budget = self.settings["crop_visual_token_budget"]
        images = [
            resize_aligned(target.scene, scene_budget * self._factor**2, self._factor),
            resize_aligned(target.crop, crop_budget * self._factor**2, self._factor),
        ]
        messages = [
            {"role": "system", "content": JSON_PROMPT.format(robot_policy=self.settings["robot_policy"])},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Scene with the target's exterior outline:"},
                    {"type": "image", "image": images[0]},
                    {"type": "text", "text": "Original RGB crop of the same target with context:"},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": "Classify only the outlined target using the frozen robot policy."},
                ],
            },
        ]
        # apply_chat_template returns modality IDs needed by Qwen3.5; keep every field.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            do_resize=False,
        )
        self._verify_inputs(inputs, images, (scene_budget, crop_budget))
        return inputs

    def _verify_inputs(self, inputs, images, budgets):
        if "input_ids" not in inputs or "image_grid_thw" not in inputs:
            raise InferenceError("processor_output", "Processor omitted input IDs or image grids.")
        grids = _tolist(inputs["image_grid_thw"])
        if not isinstance(grids, list) or len(grids) != 2:
            raise InferenceError("visual_token_count", "Expected exactly two image grids.")
        counts = []
        for grid, image, budget in zip(grids, images, budgets):
            if not isinstance(grid, (list, tuple)) or len(grid) != 3:
                raise InferenceError("visual_token_count", "Malformed processor image grid.")
            if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in grid):
                raise InferenceError("visual_token_count", "Image grid dimensions must be positive integers.")
            t, h, w = grid
            expected = (1, image.height // self._patch_size, image.width // self._patch_size)
            if (t, h, w) != expected:
                raise InferenceError("processor_resize", "Processor changed the independently aligned image dimensions.")
            if h % self._merge_size or w % self._merge_size:
                raise InferenceError("visual_token_count", "Image grid is not merge aligned.")
            tokens = t * h * w // self._merge_size**2
            if tokens > budget:
                raise InferenceError("visual_token_budget", f"Image uses {tokens} visual tokens; budget is {budget}.")
            counts.append(tokens)
        ids = _tolist(inputs["input_ids"])
        if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], list):
            raise InferenceError("processor_output", "Expected a single processor input sequence.")
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise InferenceError("processor_output", "Model configuration omitted image_token_id.")
        if sum(token == image_token_id for token in ids[0]) != sum(counts):
            raise InferenceError("visual_token_count", "Image placeholders do not match actual merged visual tokens.")
        if len(ids[0]) + self.settings["max_new_tokens"] > self.settings["context_token_limit"]:
            raise InferenceError("context_token_budget", "Input plus maximum output exceeds the context limit.")

    def predict_frame(self, frame: dict, regions: list[dict], data_root):
        """Prepare, generate and strictly parse one result per supplied region."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Qwen backend is closed.")
            results = []
            for region in regions:
                raw = ""
                try:
                    if region.get("frame_id") != frame.get("frame_id"):
                        raise InferenceError("frame_join", "Region frame_id does not match the supplied frame.")
                    target = prepare_target(frame, region, data_root)
                    inputs = self._prepare_inputs(target)
                    inputs = {key: value.to(self._device) if hasattr(value, "to") else value for key, value in inputs.items()}
                    torch = self._torch
                    with torch.cuda.device(self._device), torch.inference_mode():
                        torch.manual_seed(self.settings["seed"])
                        generated = self.model.generate(
                            **inputs,
                            do_sample=False,
                            num_beams=1,
                            num_return_sequences=1,
                            repetition_penalty=1.0,
                            max_new_tokens=self.settings["max_new_tokens"],
                            use_cache=True,
                            return_dict_in_generate=False,
                        )
                    generated_rows = _tolist(generated)
                    input_length = len(_tolist(inputs["input_ids"])[0])
                    if not isinstance(generated_rows, list) or len(generated_rows) != 1:
                        raise InferenceError("generation_output", "Expected one generated sequence.")
                    output_tokens = generated_rows[0][input_length:]
                    texts = self.processor.batch_decode(
                        [output_tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    if not isinstance(texts, list) or len(texts) != 1 or not isinstance(texts[0], str):
                        raise InferenceError("generation_output", "Expected one decoded text response.")
                    raw = texts[0]
                    if len(output_tokens) > self.settings["max_new_tokens"]:
                        raise InferenceError("generation_output", "Model exceeded the configured output limit.")
                    parsed = parse_response(raw)
                    results.append(prediction(region, self.model_key, parsed=parsed, raw_response=raw))
                except InferenceError as exc:
                    results.append(prediction(region, self.model_key, raw_response=raw, error_code=exc.code, reason=str(exc)))
                except Exception as exc:
                    results.append(
                        prediction(region, self.model_key, raw_response=raw, error_code="inference_failure", reason=f"{type(exc).__name__}: {exc}")
                    )
            return results

    def close(self):
        """Release owned resources; repeated cleanup calls are harmless."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.model = None
            self.processor = None
            gc.collect()
            torch = self._torch
            if torch is not None and hasattr(self, "_device") and torch.cuda.is_available():
                with torch.cuda.device(self._device):
                    torch.cuda.empty_cache()
