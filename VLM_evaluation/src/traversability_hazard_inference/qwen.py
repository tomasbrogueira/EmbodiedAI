"""Lazy, single-device Qwen inference over one original RGB image."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import importlib
import threading

from .configuration import (
    MODEL_SPECS, configuration_snapshot, load_policy, normalize_settings,
    prompt_text, software_versions,
)
from .preparation import load_rgb, resize_aligned
from .records import InferenceError, parse_response, prediction
from .storage import stable_fingerprint


_EXCLUDED_MODULES = ("model.visual", "lm_head")


def _tolist(value):
    return value.tolist() if hasattr(value, "tolist") else value


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


class QwenBackend:
    """Load exactly the requested checkpoint; never fall back or offload."""

    def __init__(self, model_key: str, settings: dict):
        self.model_key = model_key
        self.settings = normalize_settings(model_key, settings)
        self._policy = load_policy(self.settings)
        self._prompt = prompt_text(self._policy)
        self.model = None
        self.processor = None
        self._torch = None
        self._closed = False
        self._lock = threading.RLock()
        self.metadata = {}
        self.last_diagnostics = {}
        try:
            self._load()
        except BaseException:
            self.close()
            raise

    def _load(self):
        # Importing this module does not import any model runtime.
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        bnb = importlib.import_module("bitsandbytes")
        self._torch = torch
        requested_device = self.settings["device"]
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen 4-bit inference requires an available CUDA GPU; CPU fallback is disabled.")
        index = int(requested_device.split(":")[1])
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {requested_device}")
        self._device = torch.device(requested_device)
        spec = MODEL_SPECS[self.model_key]
        common = {"revision": spec["revision"], "local_files_only": self.settings["local_files_only"]}
        if self.settings.get("cache_dir") is not None:
            common["cache_dir"] = self.settings["cache_dir"]
        with torch.cuda.device(self._device):
            # Emulated BF16 is insufficient for the intended floating vision path.
            try:
                native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
                bf16_check = "is_bf16_supported(including_emulation=False)"
            except TypeError:
                native_bf16 = torch.cuda.get_device_capability(self._device)[0] >= 8
                bf16_check = "device_capability_major>=8 (older Torch signature)"
            dtype = torch.bfloat16 if native_bf16 else torch.float16
            device_capability = tuple(torch.cuda.get_device_capability(self._device))
            device_name = torch.cuda.get_device_name(self._device)
            quantization = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_storage=torch.uint8,
                llm_int8_skip_modules=list(_EXCLUDED_MODULES),
                llm_int8_enable_fp32_cpu_offload=False,
            )
            self.processor = transformers.AutoProcessor.from_pretrained(spec["checkpoint"], **common)
            model_class = getattr(transformers, spec["model_class"])
            self.model = model_class.from_pretrained(
                spec["checkpoint"], **common, dtype=dtype,
                device_map={"": requested_device}, quantization_config=quantization,
                attn_implementation="sdpa", use_kernels=False,
            )
            self.model.eval()
            audit = self._audit_loaded_model(torch, bnb, dtype)
        image_processor = self.processor.image_processor
        self._patch_size = int(image_processor.patch_size)
        self._merge_size = int(image_processor.merge_size)
        if self._patch_size <= 0 or self._merge_size <= 0:
            raise RuntimeError("Invalid Qwen image processor alignment settings.")
        self._factor = self._patch_size * self._merge_size
        template = getattr(self.processor, "chat_template", None)
        if template is None:
            template = getattr(self.processor.tokenizer, "chat_template", "")
        self._eos_ids = self._eos_token_ids()
        if not self._eos_ids:
            raise RuntimeError("The loaded checkpoint has no auditable EOS token ID.")
        snapshot = configuration_snapshot(self.model_key, self.settings)
        if snapshot["policy_fingerprint"] != stable_fingerprint(self._policy) or snapshot["prompt"] != self._prompt:
            raise RuntimeError("Frozen common policy changed while loading this backend.")
        self.metadata = {
            "adapter": "qwen_whole_image",
            "model_key": self.model_key,
            "checkpoint": spec["checkpoint"], "revision": spec["revision"],
            "model_class": type(self.model).__name__,
            "configuration_snapshot": deepcopy(snapshot),
            "settings_digest": stable_fingerprint(snapshot),
            "policy_sha256": snapshot["policy_sha256"],
            "policy_fingerprint": stable_fingerprint(self._policy),
            "alias_sha256": stable_fingerprint(self._policy["aliases"]),
            "prompt_sha256": hashlib.sha256(self._prompt.encode("utf-8")).hexdigest(),
            "device": requested_device, "device_name": str(device_name),
            "device_capability": list(device_capability),
            "cuda_runtime_version": str(getattr(getattr(torch, "version", None), "cuda", None)),
            "attention_implementation": "sdpa", "use_kernels": False,
            "quantization": {
                "language_bits": 4, "type": "nf4", "double_quantization": True,
                "compute_dtype": str(dtype), "native_bf16": bool(native_bf16),
                "quant_storage": str(torch.uint8),
                "native_bf16_check": bf16_check,
                "excluded_modules": list(_EXCLUDED_MODULES), "cpu_offload": False,
            },
            "processor": {
                "class": type(self.processor).__name__,
                "image_processor_class": type(image_processor).__name__,
                "tokenizer_class": type(self.processor.tokenizer).__name__,
                "image_processor_config": _portable_processor_config(_jsonable(image_processor.to_dict())),
                "chat_template_sha256": hashlib.sha256(str(template).encode("utf-8")).hexdigest(),
                "patch_size": self._patch_size, "merge_size": self._merge_size,
                "alignment_factor": self._factor, "do_resize": False,
                "max_pixels": self.settings["visual_token_budget"] * self._factor**2,
                "input_images": 1,
                "token_verification": "one_grid_product_div_merge_squared_and_image_placeholders",
            },
            "decoding": {
                "do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                "repetition_penalty": 1.0, "max_new_tokens": self.settings["max_new_tokens"],
                "context_token_limit": self.settings["context_token_limit"],
                "use_cache": True, "request_concurrency": 1, "enable_thinking": False,
                "eos_token_ids": sorted(self._eos_ids), "raw_skip_special_tokens": False,
                "seed": self.settings["seed"], "seed_applied": False,
            },
            "software_versions": software_versions(), "audit": audit,
        }

    def _audit_loaded_model(self, torch, bnb, dtype):
        quantized, language_quantized, floating_language = [], [], []
        vision_modules, quantization_states, output_head = [], {}, None
        for name, module in self.model.named_modules():
            is_quantized = isinstance(module, bnb.nn.Linear4bit)
            is_vision = name == "model.visual" or name.startswith("model.visual.")
            if is_vision:
                vision_modules.append(name)
                if is_quantized:
                    raise RuntimeError(f"Vision module was quantized despite exclusion: {name}")
            if name == "lm_head" and is_quantized:
                raise RuntimeError("The excluded lm_head was quantized.")
            if name == "lm_head":
                output_head = getattr(module, "weight", None)
            is_language = name.startswith("model.language_model.")
            if is_language and isinstance(module, torch.nn.Linear) and not is_quantized:
                floating_language.append(name)
            if not is_quantized:
                continue
            quantized.append(name)
            if not is_language:
                raise RuntimeError(f"Quantized module is outside the language model: {name}")
            language_quantized.append(name)
            weight = module.weight
            if not isinstance(weight, bnb.nn.Params4bit):
                raise RuntimeError(f"Language module weight is not a Params4bit instance: {name}")
            if weight.dtype != torch.uint8 or getattr(weight, "quant_storage", None) != torch.uint8:
                raise RuntimeError(f"Language module does not use the requested uint8 quantized storage: {name}")
            state = getattr(weight, "quant_state", None)
            if state is None or getattr(weight, "bnb_quantized", False) is not True:
                raise RuntimeError(f"Language module has no materialized 4-bit quantization state: {name}")
            quant_type = getattr(state, "quant_type", None)
            if quant_type != "nf4" or getattr(weight, "quant_type", quant_type) != "nf4":
                raise RuntimeError(f"Language module does not use NF4: {name}")
            double_quantized = getattr(state, "state2", None) is not None
            if not double_quantized:
                raise RuntimeError(f"Language module lacks double quantization: {name}")
            compute_dtype = getattr(module, "compute_dtype", None)
            if compute_dtype != dtype:
                raise RuntimeError(f"Language module has unexpected compute dtype: {name}: {compute_dtype}")
            state_devices = {}
            for prefix, quant_state in (("", state), ("state2.", state.state2)):
                for field in ("absmax", "code", "offset"):
                    tensor = getattr(quant_state, field, None)
                    if tensor is not None and hasattr(tensor, "device"):
                        placement = str(tensor.device)
                        state_devices[prefix + field] = placement
                        if placement != str(self._device):
                            raise RuntimeError(f"Quantization state {name}.{prefix}{field} is on {placement}.")
            quantization_states[name] = {
                "quant_type": quant_type, "double_quantization": double_quantized,
                "compute_dtype": str(compute_dtype), "weight_dtype": str(weight.dtype),
                "original_dtype": str(getattr(state, "dtype", None)),
                "quantization_state_devices": state_devices,
            }
        if not language_quantized:
            raise RuntimeError("No quantized language linear modules were found.")
        if floating_language:
            raise RuntimeError(f"Language linear modules remain unquantized: {floating_language}")
        if not vision_modules:
            raise RuntimeError("Expected floating-point model.visual was not found.")
        if output_head is None or not output_head.is_floating_point():
            raise RuntimeError("The excluded output head has no floating-point weight.")
        if str(output_head.device) != str(self._device):
            raise RuntimeError("The excluded output head weight is offloaded from the selected CUDA device.")
        placements, dtypes, vision_dtypes = defaultdict(list), defaultdict(list), defaultdict(list)
        vision_count = 0
        for name, parameter in self.model.named_parameters():
            placement = str(parameter.device)
            if placement != str(self._device):
                raise RuntimeError(f"Parameter {name} is on {placement}; expected {self._device} without offload.")
            placements[placement].append(name)
            dtypes[str(parameter.dtype)].append(name)
            if name.startswith("model.visual."):
                vision_count += 1
                if not parameter.is_floating_point():
                    raise RuntimeError(f"Vision parameter is not floating point: {name}")
                vision_dtypes[str(parameter.dtype)].append(name)
            if name.startswith("lm_head.") and not parameter.is_floating_point():
                raise RuntimeError(f"The excluded output head is not floating point: {name}")
        if not vision_count:
            raise RuntimeError("Expected vision parameters were not found.")
        device_map = getattr(self.model, "hf_device_map", {})
        for name, device in device_map.items():
            normalized = f"cuda:{device}" if isinstance(device, int) else str(device)
            if normalized != str(self._device):
                raise RuntimeError(f"Module {name} is dispatched to {device}; CPU/disk offload is prohibited.")
        buffers = defaultdict(list)
        if hasattr(self.model, "named_buffers"):
            for name, buffer in self.model.named_buffers():
                buffers[str(buffer.device)].append(name)
                if str(buffer.device) != str(self._device):
                    raise RuntimeError(f"Buffer {name} is on {buffer.device}; CPU/disk offload is prohibited.")
        return {
            "parameter_placement": dict(placements), "parameter_dtypes": dict(dtypes),
            "buffer_placement": dict(buffers), "hf_device_map": _jsonable(device_map),
            "vision_parameter_dtypes": dict(vision_dtypes), "quantized_modules": quantized,
            "language_quantized_modules": language_quantized,
            "floating_language_linear_modules": floating_language,
            "quantization_states": quantization_states, "vision_is_floating_point": True,
            "output_head_weight": {"device": str(output_head.device), "dtype": str(output_head.dtype)},
        }

    def _eos_token_ids(self):
        for source in (getattr(self.model, "generation_config", None),
                       getattr(self.model, "config", None), self.processor.tokenizer):
            value = getattr(source, "eos_token_id", None)
            if type(value) is int and value >= 0:
                return {value}
            elif isinstance(value, (list, tuple)):
                result = {item for item in value if type(item) is int and item >= 0}
                if result:
                    return result
        return set()

    def _prepare_inputs(self, rgb):
        image = resize_aligned(rgb, self.settings["visual_token_budget"] * self._factor**2, self._factor)
        self.last_diagnostics["preprocessing"] = {
            "original_width": rgb.width, "original_height": rgb.height,
            "processed_width": image.width, "processed_height": image.height,
            "resampling": "bicubic", "do_resize": False, "input_images": 1,
            "alignment_factor": self._factor,
        }
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "List visible concepts to avoid in this whole RGB image according to the common policy."},
            ]},
        ]
        try:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False, do_resize=False,
            )
            self._verify_inputs(inputs, image)
            return inputs
        finally:
            if image is not rgb and hasattr(image, "close"):
                image.close()

    def _verify_inputs(self, inputs, image):
        if not isinstance(inputs, dict) and not hasattr(inputs, "keys"):
            raise InferenceError("processor_output", "Processor output must be a mapping.")
        if "input_ids" not in inputs or "image_grid_thw" not in inputs:
            raise InferenceError("processor_output", "Processor omitted input IDs or image grids.")
        grids = _tolist(inputs["image_grid_thw"])
        if not isinstance(grids, list) or len(grids) != 1:
            raise InferenceError("visual_token_count", "Expected exactly one whole-image grid.")
        grid = grids[0]
        if not isinstance(grid, (list, tuple)) or len(grid) != 3 or any(type(value) is not int or value <= 0 for value in grid):
            raise InferenceError("visual_token_count", "Image grid dimensions must be three positive integers.")
        t, h, w = grid
        if (t, h, w) != (1, image.height // self._patch_size, image.width // self._patch_size):
            raise InferenceError("processor_resize", "Processor changed independently aligned image dimensions.")
        if h % self._merge_size or w % self._merge_size:
            raise InferenceError("visual_token_count", "Image grid is not merge aligned.")
        visual_tokens = t * h * w // self._merge_size**2
        if visual_tokens > self.settings["visual_token_budget"]:
            raise InferenceError("visual_token_budget", "Image exceeds the configured visual-token cap.")
        ids = _tolist(inputs["input_ids"])
        if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], list):
            raise InferenceError("processor_output", "Expected one processor input sequence.")
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if type(image_token_id) is not int:
            raise InferenceError("processor_output", "Model configuration omitted image_token_id.")
        if sum(token == image_token_id for token in ids[0]) != visual_tokens:
            raise InferenceError("visual_token_count", "Image placeholders do not match merged visual tokens.")
        self.last_diagnostics["tokens"] = {
            "input_tokens": len(ids[0]), "visual_tokens": visual_tokens,
            "nonvisual_input_tokens": len(ids[0]) - visual_tokens,
            "visual_token_budget": self.settings["visual_token_budget"],
            "context_token_limit": self.settings["context_token_limit"],
            "max_new_tokens": self.settings["max_new_tokens"],
        }
        if len(ids[0]) + self.settings["max_new_tokens"] > self.settings["context_token_limit"]:
            raise InferenceError("context_token_budget", "Input plus maximum output exceeds the context limit.")

    def _decode(self, tokens):
        texts = self.processor.batch_decode(
            [tokens], skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        if not isinstance(texts, list) or len(texts) != 1 or not isinstance(texts[0], str):
            raise InferenceError("generation_output", "Expected one decoded text response.")
        return texts[0]

    def predict_image(self, frame: dict, data_root):
        """Generate one strict versioned prediction for the supplied RGB frame."""
        with self._lock:
            raw, rgb = "", None
            self.last_diagnostics = {"frame_id": frame.get("frame_id"), "model_key": self.model_key}
            try:
                if self._closed:
                    raise InferenceError("backend_closed", "Qwen backend is closed.")
                rgb = load_rgb(frame, data_root)
                inputs = self._prepare_inputs(rgb)
                inputs = {key: value.to(self._device) if hasattr(value, "to") else value for key, value in inputs.items()}
                input_rows = _tolist(inputs["input_ids"])
                input_length = len(input_rows[0])
                torch = self._torch
                with torch.cuda.device(self._device), torch.inference_mode():
                    generated = self.model.generate(
                        **inputs, do_sample=False, num_beams=1, num_return_sequences=1,
                        repetition_penalty=1.0, max_new_tokens=self.settings["max_new_tokens"],
                        use_cache=True, return_dict_in_generate=False,
                    )
                rows = _tolist(generated)
                if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
                    raise InferenceError("generation_output", "Expected one generated sequence.")
                if rows[0][:input_length] != input_rows[0]:
                    raise InferenceError("generation_output", "Generated sequence does not preserve the input prefix.")
                suffix = rows[0][input_length:]
                raw = self._decode(suffix)
                has_eos = bool(suffix) and suffix[-1] in self._eos_ids
                self.last_diagnostics["tokens"]["output_tokens"] = len(suffix)
                self.last_diagnostics["generation"] = {
                    "terminal_eos": has_eos, "eos_token_id": suffix[-1] if has_eos else None,
                    "eos_text": self._decode(suffix[-1:]) if has_eos else None,
                    "stop_reason": "eos" if has_eos else "unverified_stop",
                }
                limit = self.settings["max_new_tokens"]
                if len(suffix) > limit:
                    raise InferenceError("generation_output", "Model exceeded the configured output limit.")
                if len(suffix) == limit and not has_eos:
                    self.last_diagnostics["generation"]["stop_reason"] = "length_limit"
                    raise InferenceError("generation_truncated", "Generation reached max_new_tokens without terminal EOS.")
                if not suffix:
                    raise InferenceError("generation_output", "Model generated no output tokens.")
                if not has_eos:
                    raise InferenceError("generation_unverified_stop", "Generation stopped without a verified terminal EOS.")
                response = self._decode(suffix[:-1]) if has_eos else raw
                self.last_diagnostics["parse_response_text"] = response
                parsed = parse_response(response, self._policy)
                self.last_diagnostics.update(parsed["diagnostics"])
                self.last_diagnostics.update(status="ok", error_code=None)
                return prediction(frame, self.model_key, prompts=parsed["prompts"], raw_response=raw)
            except Exception as error:
                code = error.code if isinstance(error, InferenceError) else "inference_failure"
                self.last_diagnostics.update(status="error", error_code=code,
                                             error_message=f"{type(error).__name__}: {error}")
                return prediction(frame, self.model_key, raw_response=raw, error_code=code)
            finally:
                if rgb is not None and hasattr(rgb, "close"):
                    rgb.close()

    def close(self):
        """Drop this backend's owned resources without global CUDA operations."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # bitsandbytes 0.50.2 Params4bit keeps a strong reference back to its
            # Linear4bit owner. Break those owned cycles so reference counting
            # can release CUDA weights without collecting other components.
            model = self.model
            try:
                if model is not None and callable(getattr(model, "named_modules", None)):
                    for _, module in model.named_modules():
                        weight = getattr(module, "weight", None)
                        if weight is not None and getattr(weight, "module", None) is module:
                            weight.module = None
            except Exception:
                # Cleanup must preserve the original error if loading produced
                # an incomplete or corrupted model object.
                pass
            finally:
                self.model = None
                self.processor = None
                self._torch = None
