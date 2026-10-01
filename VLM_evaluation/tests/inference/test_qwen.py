"""CPU adapter checks: all model libraries and weights are mocked."""

from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
import os
import subprocess
import sys

from PIL import Image
import pytest

from traversability_inference.configuration import MODEL_SPECS
from traversability_inference.qwen import QwenBackend
from traversability_inference.records import InferenceError


class FakeTensor:
    def __init__(self, values, dtype="torch.bfloat16", device="cuda:0"):
        self.values = values
        self.dtype = dtype
        self.device = device
        self.transfers = []

    def tolist(self):
        return self.values

    def to(self, device):
        self.transfers.append(str(device))
        return self

    def is_floating_point(self):
        return self.dtype in {"torch.bfloat16", "torch.float16", "torch.float32"}


class FakeLinear:
    pass


class FakeLinear4bit(FakeLinear):
    pass


@pytest.fixture
def model_dependencies(monkeypatch):
    state = SimpleNamespace(
        bf16=True,
        cuda_available=True,
        seeds=[],
        empty_cache_calls=0,
        config_calls=[],
        model_loads=[],
        processor_loads=[],
        template_calls=[],
        generation_calls=[],
        decode_calls=[],
        raw='{"label":"traversable","semantic_class":"dirt","reason":"Dry compact soil."}',
        mutate_inputs=None,
        mutate_model=None,
        generation_error=None,
        output_length=3,
        processor_config_override=None,
    )

    def empty_cache():
        state.empty_cache_calls += 1

    torch = ModuleType("torch")
    torch.bfloat16 = "torch.bfloat16"
    torch.float16 = "torch.float16"
    torch.nn = SimpleNamespace(Linear=FakeLinear)
    torch.device = str
    torch.cuda = SimpleNamespace(
        is_available=lambda: state.cuda_available,
        device_count=lambda: 1,
        device=lambda device: nullcontext(),
        is_bf16_supported=lambda: state.bf16,
        empty_cache=empty_cache,
    )
    torch.inference_mode = nullcontext
    torch.manual_seed = state.seeds.append
    bnb = ModuleType("bitsandbytes")
    bnb.nn = SimpleNamespace(Linear4bit=FakeLinear4bit)
    transformers = ModuleType("transformers")

    class FakeProcessor:
        def __init__(self):
            self.image_processor = SimpleNamespace(
                patch_size=16,
                merge_size=2,
                to_dict=lambda: state.processor_config_override or {"patch_size": 16, "merge_size": 2, "do_resize": True},
            )
            self.tokenizer = SimpleNamespace(chat_template="frozen test template")
            self.chat_template = "frozen test template"

        def apply_chat_template(self, messages, **kwargs):
            state.template_calls.append((messages, kwargs))
            images = [item["image"] for item in messages[1]["content"] if item["type"] == "image"]
            grids = [[1, image.height // 16, image.width // 16] for image in images]
            visual_count = sum(t * h * w // 4 for t, h, w in grids)
            inputs = {
                "input_ids": FakeTensor([[1, 2] + [99] * visual_count + [7]]),
                "attention_mask": FakeTensor([[1] * (visual_count + 3)]),
                "mm_token_type_ids": FakeTensor([[0, 0] + [1] * visual_count + [0]]),
                "pixel_values": FakeTensor([[0.1]]),
                "image_grid_thw": FakeTensor(grids),
            }
            state.last_inputs = inputs
            if state.mutate_inputs:
                state.mutate_inputs(inputs)
            return inputs

        def batch_decode(self, tokens, **kwargs):
            state.decode_calls.append((tokens, kwargs))
            return [state.raw]

    class FakeModel:
        def __init__(self, dtype):
            self.config = SimpleNamespace(image_token_id=99)
            self.hf_device_map = {"": "cuda:0"}
            self.modules = {
                "model.visual": SimpleNamespace(),
                "model.visual.blocks.0.fc": FakeLinear(),
                "model.language_model.layers.0.self_attn.q_proj": FakeLinear4bit(),
                "lm_head": FakeLinear(),
            }
            self.parameters = {
                "model.visual.blocks.0.fc.weight": FakeTensor([], dtype=dtype),
                "model.language_model.layers.0.self_attn.q_proj.weight": FakeTensor([], dtype="torch.uint8"),
                "lm_head.weight": FakeTensor([], dtype=dtype),
            }

        def named_modules(self):
            return self.modules.items()

        def named_parameters(self):
            return self.parameters.items()

        def eval(self):
            state.eval_called = True
            return self

        def generate(self, **kwargs):
            state.generation_calls.append(kwargs)
            if state.generation_error:
                raise state.generation_error
            return FakeTensor([kwargs["input_ids"].tolist()[0] + [3] * state.output_length])

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(checkpoint, **kwargs):
            state.model_loads.append((checkpoint, kwargs))
            model = FakeModel(kwargs["dtype"])
            if state.mutate_model:
                state.mutate_model(model)
            state.model = model
            return model

    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(checkpoint, **kwargs):
            state.processor_loads.append((checkpoint, kwargs))
            state.processor = FakeProcessor()
            return state.processor

    def quantization_config(**kwargs):
        state.config_calls.append(kwargs)
        return kwargs

    transformers.BitsAndBytesConfig = quantization_config
    transformers.AutoProcessor = FakeProcessorLoader
    transformers.Qwen3VLForConditionalGeneration = FakeModelLoader
    transformers.Qwen3_5ForConditionalGeneration = FakeModelLoader
    for name, module in (("torch", torch), ("bitsandbytes", bnb), ("transformers", transformers)):
        monkeypatch.setitem(sys.modules, name, module)

    import traversability_inference.qwen as qwen

    monkeypatch.setattr(
        qwen,
        "prepare_target",
        lambda *_: SimpleNamespace(scene=Image.new("RGB", (1800, 1200)), crop=Image.new("RGB", (160, 96))),
    )
    return state


FRAME = {"frame_id": "fixture:f", "image_path": "image.png"}
REGION = {"region_id": "fixture:r", "frame_id": "fixture:f", "mask_path": "mask.png"}


@pytest.mark.parametrize("model_key", ["qwen3_vl_4b", "qwen3_5_4b", "qwen3_5_2b"])
@pytest.mark.parametrize("bf16", [True, False])
def test_pinned_explicit_loading_quantization_and_actual_metadata(model_dependencies, model_key, bf16):
    state = model_dependencies
    state.bf16 = bf16
    backend = QwenBackend(model_key, {"cache_dir": "/configurable/cache", "local_files_only": True})
    checkpoint, arguments = state.model_loads[0]
    assert checkpoint == MODEL_SPECS[model_key]["checkpoint"]
    assert arguments["revision"] == MODEL_SPECS[model_key]["revision"]
    assert state.processor_loads[0][1]["revision"] == arguments["revision"]
    assert arguments["device_map"] == {"": "cuda:0"}
    assert arguments["local_files_only"] is True
    assert arguments["cache_dir"] == "/configurable/cache"
    dtype = "torch.bfloat16" if bf16 else "torch.float16"
    assert arguments["dtype"] == dtype
    assert state.config_calls == [{
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": dtype,
        "llm_int8_skip_modules": ["model.visual", "lm_head"],
        "llm_int8_enable_fp32_cpu_offload": False,
    }]
    assert backend.metadata["quantization"]["compute_dtype"] == dtype
    assert backend.metadata["audit"]["vision_is_floating_point"] is True
    assert backend.metadata["audit"]["language_quantized_modules"] == ["model.language_model.layers.0.self_attn.q_proj"]
    assert backend.metadata["audit"]["vision_parameter_dtypes"] == {dtype: ["model.visual.blocks.0.fc.weight"]}
    assert backend.metadata["processor"]["scene_max_pixels"] == 393216
    assert backend.metadata["processor"]["crop_max_pixels"] == 262144
    backend.close()
    backend.close()
    assert state.empty_cache_calls == 1
    assert backend.model is None and backend.processor is None


def test_greedy_limits_thinking_disabled_and_all_processor_inputs_preserved(model_dependencies):
    state = model_dependencies
    backend = QwenBackend("qwen3_5_4b", {})
    result = backend.predict_frame(FRAME, [REGION], "/unused/root")
    assert result[0]["region_id"] == REGION["region_id"]
    assert result[0]["label"] == "traversable"
    assert result[0]["raw_response"] == state.raw
    assert result[0]["status"] == "ok"
    messages, template_arguments = state.template_calls[0]
    assert template_arguments == {
        "tokenize": True, "add_generation_prompt": True, "return_dict": True,
        "return_tensors": "pt", "enable_thinking": False, "do_resize": False,
    }
    assert backend.settings["robot_policy"] in messages[0]["content"]
    images = [item["image"] for item in messages[1]["content"] if item["type"] == "image"]
    assert images[0].width * images[0].height <= 393216
    assert images[1].width * images[1].height <= 262144
    assert images[0].size != images[1].size  # The crop is independently sized.
    generation = state.generation_calls[0]
    for key in state.last_inputs:
        assert generation[key] is state.last_inputs[key]
        assert generation[key].transfers == ["cuda:0"]
    assert generation["mm_token_type_ids"] is state.last_inputs["mm_token_type_ids"]
    assert generation["do_sample"] is False
    assert generation["num_beams"] == 1
    assert generation["num_return_sequences"] == 1
    assert generation["max_new_tokens"] == 128
    assert generation["return_dict_in_generate"] is False
    assert state.seeds == [0]
    assert state.decode_calls == [([[3, 3, 3]], {"skip_special_tokens": True, "clean_up_tokenization_spaces": False})]
    backend.close()


def test_qwen_families_use_identical_policy_and_prompt(model_dependencies):
    state = model_dependencies
    for key in ("qwen3_vl_4b", "qwen3_5_4b", "qwen3_5_2b"):
        backend = QwenBackend(key, {})
        backend.predict_frame(FRAME, [REGION], "/unused/root")
        backend.close()
    prompts = [messages[0]["content"] for messages, _ in state.template_calls]
    assert len(set(prompts)) == 1
    assert len(state.model_loads) == 3


@pytest.mark.parametrize("raw", [
    "```json\n{\"label\":\"unknown\",\"semantic_class\":null,\"reason\":\"unclear\"}\n```",
    '{"label":"traversable","label":"unknown","semantic_class":"dirt","reason":"dry"}',
    '{"label":"traversable","semantic_class":4,"reason":"dry"}',
])
def test_strict_parser_errors_preserve_raw_response(model_dependencies, raw):
    state = model_dependencies
    state.raw = raw
    backend = QwenBackend("qwen3_vl_4b", {})
    result = backend.predict_frame(FRAME, [REGION], "/unused/root")[0]
    assert result["label"] == "unknown"
    assert result["status"] == "error"
    assert result["error_code"]
    assert result["raw_response"] == raw
    backend.close()


@pytest.mark.parametrize("mutation,expected_code", [
    (lambda inputs: inputs["image_grid_thw"].values[0].__setitem__(1, 400), "processor_resize"),
    (lambda inputs: inputs["input_ids"].values[0].append(99), "visual_token_count"),
    (lambda inputs: inputs["input_ids"].values[0].extend([5] * 2048), "context_token_budget"),
    (lambda inputs: inputs.pop("image_grid_thw"), "processor_output"),
])
def test_actual_processor_dimensions_tokens_and_context_are_checked(model_dependencies, mutation, expected_code):
    state = model_dependencies
    state.mutate_inputs = mutation
    backend = QwenBackend("qwen3_5_4b", {})
    result = backend.predict_frame(FRAME, [REGION], "/unused/root")[0]
    assert result["status"] == "error"
    assert result["error_code"] == expected_code
    assert result["label"] == "unknown"
    assert not state.generation_calls
    backend.close()


@pytest.mark.parametrize("mutation", [
    lambda model: model.modules.__setitem__("model.visual.blocks.0.fc", FakeLinear4bit()),
    lambda model: model.modules.__setitem__("lm_head", FakeLinear4bit()),
    lambda model: model.modules.__setitem__("model.language_model.layers.0.self_attn.q_proj", FakeLinear()),
    lambda model: setattr(model.parameters["model.visual.blocks.0.fc.weight"], "dtype", "torch.uint8"),
    lambda model: setattr(model.parameters["lm_head.weight"], "device", "cpu"),
    lambda model: model.hf_device_map.__setitem__("model.visual", "disk"),
])
def test_audit_refuses_quantized_vision_floating_language_or_offload(model_dependencies, mutation):
    state = model_dependencies
    state.mutate_model = mutation
    with pytest.raises(RuntimeError):
        QwenBackend("qwen3_5_4b", {})
    assert len(state.model_loads) == 1  # Loading never switches to the 2B fallback.
    assert state.empty_cache_calls == 1


def test_generation_failure_and_bad_frame_join_return_complete_errors(model_dependencies):
    state = model_dependencies
    state.generation_error = RuntimeError("synthetic out of memory")
    backend = QwenBackend("qwen3_vl_4b", {})
    wrong = {**REGION, "region_id": "fixture:bad", "frame_id": "fixture:other"}
    result = backend.predict_frame(FRAME, [wrong, REGION], "/unused/root")
    assert len(result) == 2
    assert [record["region_id"] for record in result] == ["fixture:bad", "fixture:r"]
    assert [record["error_code"] for record in result] == ["frame_join", "inference_failure"]
    assert all(record["status"] == "error" and record["label"] == "unknown" for record in result)
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.predict_frame(FRAME, [REGION], "/unused/root")


def test_output_limit_failure_preserves_decoded_text(model_dependencies):
    state = model_dependencies
    state.output_length = 129
    backend = QwenBackend("qwen3_vl_4b", {})
    result = backend.predict_frame(FRAME, [REGION], "/unused/root")[0]
    assert result["error_code"] == "generation_output"
    assert result["raw_response"] == state.raw
    backend.close()


def test_target_preparation_failure_is_one_diagnostic_prediction(model_dependencies, monkeypatch):
    state = model_dependencies
    backend = QwenBackend("qwen3_vl_4b", {})

    def failing_preparation(*_):
        raise InferenceError("mask_empty", "Synthetic fixture mask is empty")

    monkeypatch.setattr("traversability_inference.qwen.prepare_target", failing_preparation)
    result = backend.predict_frame(FRAME, [REGION], "/unused/root")
    assert len(result) == 1
    assert result[0]["region_id"] == REGION["region_id"]
    assert result[0]["error_code"] == "mask_empty"
    assert result[0]["label"] == "unknown"
    assert not state.generation_calls
    backend.close()


def test_processor_metadata_excludes_machine_local_paths(model_dependencies):
    state = model_dependencies
    state.processor_config_override = {
        "_name_or_path": "/local/cache/model", "patch_size": 16,
        "nested": {"cache_dir": "/other/cache", "merge_size": 2},
    }
    backend = QwenBackend("qwen3_vl_4b", {})
    assert backend.metadata["processor"]["image_processor_config"] == {"patch_size": 16, "nested": {"merge_size": 2}}
    backend.close()


def test_no_cuda_gives_actionable_error_and_never_loads_weights(model_dependencies):
    state = model_dependencies
    state.cuda_available = False
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        QwenBackend("qwen3_5_4b", {})
    assert not state.model_loads


def test_import_is_free_of_model_libraries_and_downloads():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import traversability_inference; import traversability_inference.qwen; "
         "assert not {'torch', 'transformers', 'bitsandbytes'} & set(sys.modules)"],
        env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
