"""Explicit CPU fixtures: mocked model libraries; no CUDA, PIL or downloads."""

from contextlib import nullcontext
from copy import deepcopy
import gc
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from traversability_hazard_inference.configuration import MODEL_SPECS
from traversability_hazard_inference.qwen import QwenBackend
from traversability_hazard_inference.records import InferenceError


class FakeTensor:
    def __init__(self, values, dtype="torch.bfloat16", device="cuda:0"):
        self.values, self.dtype, self.device = values, dtype, device
        self.transfers = []

    def tolist(self):
        return self.values

    def to(self, device):
        self.transfers.append(str(device))
        return self

    def is_floating_point(self):
        return self.dtype in {"torch.bfloat16", "torch.float16", "torch.float32"}


class FakeImage:
    def __init__(self, width=1800, height=1200):
        self.width, self.height = width, height
        self.mode, self.closed = "RGB", False

    def close(self):
        self.closed = True


class FakeLinear:
    pass


class FakeParams4bit(FakeTensor):
    pass


class FakeLinear4bit(FakeLinear):
    def __init__(self, dtype="torch.bfloat16"):
        nested = SimpleNamespace(absmax=FakeTensor([]), code=FakeTensor([]))
        state = SimpleNamespace(quant_type="nf4", state2=nested,
                                dtype=dtype, absmax=FakeTensor([]), code=FakeTensor([]),
                                offset=FakeTensor([]))
        self.weight = FakeParams4bit([], dtype="torch.uint8")
        self.weight.quant_state = state
        self.weight.bnb_quantized = True
        self.weight.quant_type = "nf4"
        self.weight.quant_storage = "torch.uint8"
        self.weight.module = self
        self.compute_dtype = dtype


def fixture_runtime():
    state = SimpleNamespace(
        native_bf16=True, cuda_available=True, bf16_calls=[],
        model_loads=[], processor_loads=[], quantization_calls=[], template_calls=[],
        generation_calls=[], decode_calls=[], images=[], resized=[],
        raw='{"prompts":["person","puddle"]}', output_tokens=[3, 4, 2],
        mutate_inputs=None, mutate_model=None, mutate_processor=None, generation_error=None,
        generated_override=None, decoded_override=None,
    )
    torch = ModuleType("torch")
    torch.bfloat16, torch.float16 = "torch.bfloat16", "torch.float16"
    torch.uint8 = "torch.uint8"
    torch.version = SimpleNamespace(cuda="fixture-CUDA-runtime")
    torch.device = str
    torch.nn = SimpleNamespace(Linear=FakeLinear)
    torch.inference_mode = nullcontext

    def forbidden(*args, **kwargs):
        raise AssertionError("Global CUDA cache cleanup or RNG mutation is forbidden.")

    def bf16_supported(**kwargs):
        state.bf16_calls.append(kwargs)
        return state.native_bf16

    torch.cuda = SimpleNamespace(
        is_available=lambda: state.cuda_available, device_count=lambda: 1,
        device=lambda device: nullcontext(), is_bf16_supported=bf16_supported,
        get_device_capability=lambda *_: (8, 0) if state.native_bf16 else (7, 5), empty_cache=forbidden,
        get_device_name=lambda *_: "explicitly mocked GPU",
    )
    torch.manual_seed = forbidden
    bnb = ModuleType("bitsandbytes")
    bnb.nn = SimpleNamespace(Linear4bit=FakeLinear4bit, Params4bit=FakeParams4bit)
    transformers = ModuleType("transformers")

    class FakeProcessor:
        def __init__(self):
            self.image_processor = SimpleNamespace(
                patch_size=16, merge_size=2,
                to_dict=lambda: {"patch_size": 16, "merge_size": 2, "_name_or_path": "/machine/cache"},
            )
            self.tokenizer = SimpleNamespace(chat_template="fixture template", eos_token_id=2)
            self.chat_template = "fixture template"

        def apply_chat_template(self, messages, **kwargs):
            state.template_calls.append((messages, kwargs))
            images = [item["image"] for item in messages[1]["content"] if item["type"] == "image"]
            grids = [[1, image.height // 16, image.width // 16] for image in images]
            count = sum(t * h * w // 4 for t, h, w in grids)
            inputs = {
                "input_ids": FakeTensor([[1] + [99] * count + [7]]),
                "attention_mask": FakeTensor([[1] * (count + 2)]),
                "mm_token_type_ids": FakeTensor([[0] + [1] * count + [0]]),
                "pixel_values": FakeTensor([[0.1]]), "image_grid_thw": FakeTensor(grids),
            }
            state.last_inputs = inputs
            if state.mutate_inputs:
                state.mutate_inputs(inputs)
            return inputs

        def batch_decode(self, tokens, **kwargs):
            state.decode_calls.append((tokens, kwargs))
            if state.decoded_override is not None:
                return state.decoded_override(tokens)
            return [state.raw + ("<|im_end|>" if tokens[0] and tokens[0][-1] == 2 else "")]

    class FakeModel:
        def __init__(self, dtype):
            self.config = SimpleNamespace(image_token_id=99, eos_token_id=2)
            self.generation_config = SimpleNamespace(eos_token_id=[2])
            self.hf_device_map = {"": "cuda:0"}
            linear = FakeLinear4bit(dtype)
            self.modules = {
                "model.visual": SimpleNamespace(), "model.visual.fc": FakeLinear(),
                "model.language_model.layers.0.in_proj": linear, "lm_head": FakeLinear(),
            }
            self.parameters = {
                "model.visual.fc.weight": FakeTensor([], dtype=dtype),
                "model.language_model.layers.0.in_proj.weight": linear.weight,
                "lm_head.weight": FakeTensor([], dtype=dtype),
            }
            self.modules["lm_head"].weight = self.parameters["lm_head.weight"]
            self.buffers = {}

        def named_modules(self):
            return self.modules.items()

        def named_parameters(self):
            return self.parameters.items()

        def named_buffers(self):
            return self.buffers.items()

        def eval(self):
            state.eval_called = True
            return self

        def generate(self, **kwargs):
            state.generation_calls.append(kwargs)
            if state.generation_error:
                raise state.generation_error
            if state.generated_override is not None:
                return FakeTensor(state.generated_override)
            output_tokens = state.output_tokens
            for index, token in enumerate(output_tokens):
                if token in kwargs["eos_token_id"]:
                    output_tokens = output_tokens[:index + 1]
                    break
            return FakeTensor([kwargs["input_ids"].tolist()[0] + output_tokens])

    def model_loader(class_name):
        class Loader:
            @staticmethod
            def from_pretrained(checkpoint, **kwargs):
                state.model_loads.append((class_name, checkpoint, kwargs))
                model = FakeModel(kwargs["dtype"])
                if state.mutate_model:
                    state.mutate_model(model)
                state.model = model
                return model
        return Loader

    class ProcessorLoader:
        @staticmethod
        def from_pretrained(checkpoint, **kwargs):
            state.processor_loads.append((checkpoint, kwargs))
            state.processor = FakeProcessor()
            if state.mutate_processor:
                state.mutate_processor(state.processor)
            return state.processor

    def quantization_config(**kwargs):
        state.quantization_calls.append(kwargs)
        return kwargs

    transformers.AutoProcessor = ProcessorLoader
    transformers.BitsAndBytesConfig = quantization_config
    transformers.Qwen3VLForConditionalGeneration = model_loader("Qwen3VLForConditionalGeneration")
    transformers.Qwen3_5ForConditionalGeneration = model_loader("Qwen3_5ForConditionalGeneration")

    def load_rgb(frame, data_root):
        image = FakeImage()
        state.images.append(image)
        state.loaded_frame = deepcopy(frame)
        return image

    def resize(image, max_pixels, factor):
        scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
        result = FakeImage(max(factor, math.floor(image.width * scale / factor) * factor),
                           max(factor, math.floor(image.height * scale / factor) * factor))
        state.resized.append(result)
        return result

    return state, {"torch": torch, "transformers": transformers, "bitsandbytes": bnb}, load_rgb, resize


FRAME = {
    "frame_id": "fixture:whole-image", "image_path": "fixture/rgb.png",
    "source": "fixture", "timestamp_s": None,
}


class QwenCPUFixtureTests(unittest.TestCase):
    def setUp(self):
        self.state, self.modules, load, resize = fixture_runtime()
        self.patches = [patch.dict(sys.modules, self.modules),
                        patch("traversability_hazard_inference.qwen.load_rgb", load),
                        patch("traversability_hazard_inference.qwen.resize_aligned", resize)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.backends = []
        self.addCleanup(self.close_all)

    def close_all(self):
        for backend in self.backends:
            backend.close()

    def backend(self, key="qwen3_5_4b", **settings):
        backend = QwenBackend(key, dict(local_files_only=True, fixture=True, **settings))
        self.backends.append(backend)
        return backend

    def test_all_pinned_models_explicit_quantization_and_native_dtypes(self):
        # Include the drive on Windows so this is an absolute external cache.
        cache_dir = str((Path(ROOT.anchor) / "external" / "cache").resolve())
        for native in (True, False):
            for key in ("qwen3_vl_4b", "qwen3_5_4b", "qwen3_5_2b"):
                with self.subTest(model=key, native_bf16=native):
                    self.state.native_bf16 = native
                    backend = self.backend(key, cache_dir=cache_dir)
                    cls, checkpoint, kwargs = self.state.model_loads[-1]
                    spec = MODEL_SPECS[key]
                    self.assertEqual((cls, checkpoint, kwargs["revision"]),
                                     (spec["model_class"], spec["checkpoint"], spec["revision"]))
                    self.assertEqual(kwargs["device_map"], {"": "cuda:0"})
                    self.assertTrue(kwargs["local_files_only"])
                    self.assertEqual(kwargs["cache_dir"], cache_dir)
                    self.assertEqual(self.state.processor_loads[-1][1]["cache_dir"], cache_dir)
                    self.assertEqual(kwargs["attn_implementation"], "sdpa")
                    self.assertFalse(kwargs["use_kernels"])
                    dtype = "torch.bfloat16" if native else "torch.float16"
                    self.assertEqual(kwargs["dtype"], dtype)
                    self.assertEqual(self.state.quantization_calls[-1], {
                        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": dtype,
                        "bnb_4bit_quant_storage": "torch.uint8",
                        "llm_int8_skip_modules": ["model.visual", "lm_head"],
                        "llm_int8_enable_fp32_cpu_offload": False,
                    })
                    self.assertEqual(self.state.bf16_calls[-1], {"including_emulation": False})
                    audit = backend.metadata["audit"]
                    self.assertTrue(audit["vision_is_floating_point"])
                    actual = next(iter(audit["quantization_states"].values()))
                    self.assertEqual(actual["quant_type"], "nf4")
                    self.assertTrue(actual["double_quantization"])
                    self.assertEqual(actual["compute_dtype"], dtype)
                    self.assertNotIn("_name_or_path", backend.metadata["processor"]["image_processor_config"])
                    self.assertEqual(backend.metadata["processor"]["max_pixels"], 1024 * 32**2)
                    backend.close()
                    backend.close()
                    self.assertIsNone(backend.model)
                    self.assertIsNone(backend.processor)

    def test_one_image_common_prompt_no_reference_leakage_and_all_inputs_preserved(self):
        backend = self.backend()
        metadata = deepcopy(backend.metadata)
        frame = dict(FRAME, present_concepts=["SENTINEL_REFERENCE_PERSON"],
                     reference_mask_path="SENTINEL_REFERENCE_MASK", source="SENTINEL_SOURCE")
        result = backend.predict_image(frame, "/unused")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["prompts"], ["person", "puddle"])
        self.assertEqual(result["raw_response"], self.state.raw + "<|im_end|>")
        self.assertEqual(result["frame_id"], frame["frame_id"])
        self.assertEqual(set(result), {"task_id", "schema_version", "frame_id", "model_key",
                                      "prompts", "raw_response", "status", "error_code"})
        messages, kwargs = self.state.template_calls[-1]
        self.assertEqual(kwargs, dict(tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt", enable_thinking=False, do_resize=False))
        images = [item["image"] for item in messages[1]["content"] if item["type"] == "image"]
        self.assertEqual(len(images), 1)
        text = messages[0]["content"] + messages[1]["content"][-1]["text"]
        self.assertNotIn("SENTINEL", text)
        for concept in backend._policy["canonical_prompts"]:
            self.assertIn(concept, text)
        generation = self.state.generation_calls[-1]
        for key, value in self.state.last_inputs.items():
            self.assertIs(generation[key], value)
            self.assertEqual(value.transfers, ["cuda:0"])
        self.assertFalse(generation["do_sample"])
        self.assertEqual(generation["max_new_tokens"], 256)
        self.assertEqual(generation["num_beams"], 1)
        self.assertEqual(backend.metadata, metadata)
        self.assertEqual(backend.last_diagnostics["tokens"]["output_tokens"], 3)
        self.assertLessEqual(backend.last_diagnostics["tokens"]["visual_tokens"], 1024)
        self.assertEqual(backend.last_diagnostics["preprocessing"]["original_width"], 1800)
        self.assertTrue(all(image.closed for image in self.state.images + self.state.resized))
        self.assertTrue(all(not args[1]["skip_special_tokens"] for args in self.state.decode_calls))

    def test_policy_identical_across_model_families(self):
        for key in ("qwen3_vl_4b", "qwen3_5_4b", "qwen3_5_2b"):
            self.backend(key).predict_image(FRAME, "/unused")
        self.assertEqual(len({call[0][0]["content"] for call in self.state.template_calls}), 1)

    def test_older_bf16_signature_falls_back_to_native_capability(self):
        self.state.native_bf16 = False
        self.modules["torch"].cuda.is_bf16_supported = lambda: True
        backend = self.backend()
        self.assertEqual(self.state.model_loads[-1][2]["dtype"], "torch.float16")
        self.assertIn("device_capability", backend.metadata["quantization"]["native_bf16_check"])

    def test_generation_and_tokenizer_eos_are_both_explicit_stops(self):
        self.state.mutate_model = lambda model: setattr(model.generation_config, "eos_token_id", [5])
        backend = self.backend()
        result = backend.predict_image(FRAME, "/unused")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.state.generation_calls[-1]["eos_token_id"], [2, 5])
        self.assertEqual(backend.metadata["decoding"]["eos_token_ids"], [2, 5])

    def test_qwen35_document_and_chat_eos_stop_at_chat_turn_end(self):
        # The pinned 4B checkpoint has no generation_config.json. Transformers
        # synthesizes document EOS 248044 from text_config; its tokenizer uses
        # chat EOS 248046. Reproduce the suffix that previously ran past im_end.
        def pinned_eos(model):
            model.generation_config.eos_token_id = 248044
            model.config.eos_token_id = None
            model.config.text_config = SimpleNamespace(eos_token_id=248044)

        self.state.mutate_model = pinned_eos
        self.state.mutate_processor = lambda processor: setattr(processor.tokenizer, "eos_token_id", 248046)
        self.state.output_tokens = [3, 248046, 8, 248044]
        token_text = {3: '{"prompts":[]}', 248046: "<|im_end|>", 8: "\n", 248044: "<|endoftext|>"}
        self.state.decoded_override = lambda rows: ["".join(token_text[token] for token in rows[0])]
        backend = self.backend()
        result = backend.predict_image(FRAME, "/unused")
        self.assertEqual(self.state.generation_calls[-1]["eos_token_id"], [248044, 248046])
        self.assertEqual(backend.metadata["decoding"]["eos_token_ids"], [248044, 248046])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["prompts"], [])
        self.assertEqual(result["raw_response"], '{"prompts":[]}<|im_end|>')
        self.assertEqual(backend.last_diagnostics["parse_response_text"], '{"prompts":[]}')
        self.assertEqual(backend.last_diagnostics["generation"]["eos_token_id"], 248046)
        self.assertEqual(backend.last_diagnostics["tokens"]["output_tokens"], 2)

    def test_nested_text_eos_is_audited_even_without_generation_config(self):
        def nested_eos(model):
            model.generation_config = None
            model.config.eos_token_id = None
            model.config.text_config = SimpleNamespace(eos_token_id=[5, True, -1, "6"])

        self.state.mutate_model = nested_eos
        backend = self.backend()
        self.assertEqual(backend.metadata["decoding"]["eos_token_ids"], [2, 5])

    def test_cap_without_eos_is_error_even_when_decoded_json_is_valid(self):
        self.state.output_tokens = [3] * 256
        backend = self.backend()
        result = backend.predict_image(FRAME, "/unused")
        self.assertEqual(result["error_code"], "generation_truncated")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["raw_response"], self.state.raw)
        self.assertEqual(backend.last_diagnostics["generation"]["stop_reason"], "length_limit")

    def test_terminal_eos_at_cap_is_complete(self):
        self.state.output_tokens = [3] * 255 + [2]
        result = self.backend().predict_image(FRAME, "/unused")
        self.assertEqual(result["status"], "ok")

    def test_stop_without_eos_below_cap_is_explicit_error(self):
        self.state.output_tokens = [3, 4]
        result = self.backend().predict_image(FRAME, "/unused")
        self.assertEqual(result["error_code"], "generation_unverified_stop")
        self.assertEqual(result["raw_response"], self.state.raw)

    def test_over_budget_output_and_empty_output_are_errors(self):
        for tokens in ([3] * 256 + [2], []):
            with self.subTest(tokens=len(tokens)):
                self.state.output_tokens = tokens
                result = self.backend().predict_image(FRAME, "/unused")
                self.assertEqual(result["error_code"], "generation_output")

    def test_only_verified_terminal_eos_removed_no_hidden_special_tokens(self):
        self.state.decoded_override = lambda tokens: ["<think>" + self.state.raw +
                                                      ("<|im_end|>" if tokens[0][-1:] == [2] else "")]
        result = self.backend().predict_image(FRAME, "/unused")
        self.assertEqual(result["status"], "error")
        self.assertIn("<think>", result["raw_response"])

    def test_strict_parse_errors_retain_full_raw_and_valid_empty_is_ok(self):
        for raw in ('```json\n{"prompts":[]}\n```', '{"prompts":[],"other":1}',
                    '{"prompts":[""]}', '{"prompts":null}', '{"prompts":[],"prompts":[]}',
                    '{"prompts":[]}<|im_end|>\n'):
            with self.subTest(raw=raw):
                self.state.raw = raw
                result = self.backend().predict_image(FRAME, "/unused")
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["raw_response"], raw + "<|im_end|>")
                self.assertTrue(result["error_code"])
        self.state.raw = '{"prompts":[]}'
        result = self.backend().predict_image(FRAME, "/unused")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["prompts"], [])

    def test_processor_grid_count_placeholders_resize_and_context_errors(self):
        mutations = [
            (lambda inputs: inputs["image_grid_thw"].values.append([1, 2, 2]), "visual_token_count"),
            (lambda inputs: inputs["image_grid_thw"].values[0].__setitem__(1, 400), "processor_resize"),
            (lambda inputs: inputs["input_ids"].values[0].append(99), "visual_token_count"),
            (lambda inputs: inputs["input_ids"].values[0].extend([5] * 2048), "context_token_budget"),
            (lambda inputs: inputs.pop("image_grid_thw"), "processor_output"),
        ]
        for mutation, code in mutations:
            with self.subTest(code=code):
                self.state.mutate_inputs = mutation
                previous = len(self.state.generation_calls)
                result = self.backend().predict_image(FRAME, "/unused")
                self.assertEqual(result["error_code"], code)
                self.assertEqual(len(self.state.generation_calls), previous)

    def test_generation_failure_bad_prefix_and_closed_backend_return_errors(self):
        backend = self.backend()
        self.state.generation_error = RuntimeError("synthetic OOM")
        result = backend.predict_image(FRAME, "/unused")
        self.assertEqual(result["error_code"], "inference_failure")
        self.assertIn("synthetic OOM", backend.last_diagnostics["error_message"])
        self.state.generation_error = None
        self.state.generated_override = [[42, 2]]
        result = backend.predict_image(FRAME, "/unused")
        self.assertEqual(result["error_code"], "generation_output")
        backend.close()
        result = backend.predict_image(FRAME, "/unused")
        self.assertEqual(result["error_code"], "backend_closed")

    def test_rgb_preparation_failure_is_one_error(self):
        backend = self.backend()
        with patch("traversability_hazard_inference.qwen.load_rgb",
                   side_effect=InferenceError("image_hash_mismatch", "Fixture corruption")):
            result = backend.predict_image(FRAME, "/unused")
        self.assertEqual(result["error_code"], "image_hash_mismatch")
        self.assertFalse(self.state.generation_calls)

    def test_quantization_and_device_audits_fail_without_model_switch_or_cleanup(self):
        key = "model.language_model.layers.0.in_proj"
        mutations = [
            lambda model: model.modules.__setitem__("model.visual.fc", FakeLinear4bit()),
            lambda model: model.modules.__setitem__("lm_head", FakeLinear4bit()),
            lambda model: model.modules.__setitem__(key, FakeLinear()),
            lambda model: setattr(model.modules[key].weight, "quant_state", None),
            lambda model: setattr(model.modules[key], "weight", FakeTensor([])),
            lambda model: setattr(model.modules[key].weight, "bnb_quantized", False),
            lambda model: setattr(model.modules[key].weight, "dtype", "torch.float32"),
            lambda model: setattr(model.modules[key].weight, "quant_storage", "torch.float16"),
            lambda model: setattr(model.modules[key].weight.quant_state, "quant_type", "fp4"),
            lambda model: setattr(model.modules[key].weight.quant_state, "state2", None),
            lambda model: setattr(model.modules[key], "compute_dtype", "torch.float16"),
            lambda model: setattr(model.modules[key].weight.quant_state.absmax, "device", "cpu"),
            lambda model: setattr(model.parameters["model.visual.fc.weight"], "dtype", "torch.uint8"),
            lambda model: setattr(model.parameters["lm_head.weight"], "device", "cpu"),
            lambda model: setattr(model.modules["lm_head"], "weight", FakeTensor([], dtype="torch.uint8")),
            lambda model: model.buffers.__setitem__("rotary", FakeTensor([], device="cpu")),
            lambda model: model.hf_device_map.__setitem__("model.visual", "disk"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.state.mutate_model = mutation
                before = len(self.state.model_loads)
                with self.assertRaises(RuntimeError):
                    self.backend()
                self.assertEqual(len(self.state.model_loads), before + 1)
                self.assertEqual(self.state.model_loads[-1][1], MODEL_SPECS["qwen3_5_4b"]["checkpoint"])

    def test_no_cuda_or_missing_selected_device_never_loads_weights(self):
        self.state.cuda_available = False
        with self.assertRaisesRegex(RuntimeError, "CUDA GPU"):
            self.backend()
        self.assertFalse(self.state.model_loads)
        self.state.cuda_available = True
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.backend(device="cuda:1")
        self.assertFalse(self.state.model_loads)

    def test_close_breaks_only_owned_quantized_weight_cycles_without_global_gc(self):
        backend = self.backend()
        key = "model.language_model.layers.0.in_proj"
        module = backend.model.modules[key]
        weight = module.weight
        model_ref, module_ref, weight_ref = weakref.ref(backend.model), weakref.ref(module), weakref.ref(weight)
        # The fake loader retains a reference for assertions; remove it so the
        # backend alone owns this fixture model and its quantized weight cycle.
        self.state.model = None
        del module, weight
        previously_enabled = gc.isenabled()
        gc.disable()
        try:
            with patch("gc.collect", side_effect=AssertionError("Global GC is forbidden.")):
                backend.close()
                backend.close()
            self.assertIsNone(model_ref())
            self.assertIsNone(module_ref())
            self.assertIsNone(weight_ref())
        finally:
            if previously_enabled:
                gc.enable()

    def test_close_preserves_foreign_weight_backrefs_and_handles_partial_models(self):
        backend = self.backend()
        weight = backend.model.modules["model.language_model.layers.0.in_proj"].weight
        foreign_owner = object()
        weight.module = foreign_owner
        backend.close()
        self.assertIs(weight.module, foreign_owner)
        partial = self.backend()
        partial.model = SimpleNamespace(named_modules=lambda: (_ for _ in ()).throw(RuntimeError("partial fixture")))
        partial.close()
        self.assertIsNone(partial.model)

    def test_import_has_no_model_runtime_side_effects(self):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import sys; import traversability_hazard_inference; "
             "import traversability_hazard_inference.qwen; "
             "assert not {'torch','transformers','bitsandbytes'} & set(sys.modules)"],
            env=env, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
