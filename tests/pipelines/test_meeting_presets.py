"""Meeting presets through the actual CPU SAM reader and semantic adapters.

These fixtures prove prompt handoff, role mapping and shared record contracts.
They do not execute SAM weights or establish segmentation quality or GPU speed.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "VLM_evaluation" / "src"))

from pipeline_common.contracts import FramePacket, SemanticFrame, validate_semantic
from pipeline_common.io import file_sha256, rgb_sha256
from pipelines.fixed_hazards import create_adapter as create_vocabulary_adapter
from pipelines.ground_surface import create_adapter as create_surface_adapter
from traversability_hazard_segmentation import load_segmenter
from traversability_hazard_segmentation.sam3_adapter import Sam3Segmenter


HISTORICAL_HAZARDS = [
    "tree", "pole", "water", "vehicle", "building", "log", "person", "fence",
    "bush", "barrier", "mud", "rubble", "cup", "bottle", "chair", "dog",
]
SURFACE_PHRASES = [
    "floor", "ground", "path", "sidewalk", "pavement", "asphalt", "concrete",
    "dirt road", "gravel",
]
ADDED_HAZARDS = ["stairs", "step", "wall", "rock", "cable", "bicycle", "bench"]
RICH_POLICY_PATH = "configs/pipelines/fixed_vocabulary_surface_hazards_v1.policy.json"


def read_preset(name):
    return json.loads((ROOT / "configs" / "pipelines" / name).read_text(encoding="utf-8"))


class RecordingProcessor:
    """Record the real SAM reader's image and text calls with synthetic masks."""

    def __init__(self, detections=1):
        self.detections = detections
        self.images = []
        self.prompts = []
        self.reset_count = 0

    def set_image(self, image):
        self.images.append(np.array(image))
        self.shape = (image.height, image.width)
        return {"image_features": "explicit_cpu_fixture"}

    def reset_all_prompts(self, state):
        self.reset_count += 1
        state.pop("masks", None)
        state.pop("scores", None)

    def set_text_prompt(self, *, state, prompt):
        self.prompts.append(prompt)
        masks = np.zeros((self.detections, *self.shape), dtype=np.bool_)
        if self.detections:
            masks[:, 1:, 1:] = True
        state.update(masks=masks, scores=np.full(self.detections, 0.8))
        return state


class MeetingPresetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="meeting-presets-")
        self.addCleanup(self.temporary.cleanup)
        self.rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        image_path = Path(self.temporary.name) / "processed.png"
        Image.fromarray(self.rgb).save(image_path)
        self.frame = FramePacket(
            sequence_id="meeting-fixture-sequence", frame_id="frame-0001",
            timestamp_ns=None, timestamp_provenance={"clock": "unknown", "kind": "unknown"},
            rgb=self.rgb, image_path=image_path, encoded_file_sha256=file_sha256(image_path),
            decoded_rgb_sha256=rgb_sha256(self.rgb), processed_grid_id="shared-processed-grid",
            source_rgb_identity={"fixture": True, "frame_id": "source-0001"},
            source_to_processed={"recipe": "synthetic_identity_v1", "processed_shape": [4, 6]},
        )

    def surface_observation(self, preset, *, detections=1):
        processor = RecordingProcessor(detections)
        settings = {**preset["pipeline"]["sam3"], "fixture": True}
        segmenter = Sam3Segmenter(settings, processor=processor)
        adapter = create_surface_adapter({
            **deepcopy(preset["pipeline"]), "fixture": True, "_segmenter": segmenter,
        })
        self.addCleanup(adapter.close)
        result = adapter.observe(self.frame)
        validate_semantic(result, self.frame)
        return result, processor

    def test_floor_preset_retains_indoor_model_settings_and_declares_meeting_choice(self):
        indoor = read_preset("ground_surface_indoor.json")
        floor = read_preset("ground_surface_floor.json")
        self.assertEqual(floor["pipeline_id"], "ground_surface")
        self.assertEqual(floor["pipeline"]["prompt"], "floor")
        self.assertEqual(floor["pipeline"]["config_id"], "ground_surface_indoor_v1")
        self.assertEqual(floor["pipeline"]["prompt_selection"], {
            "frozen": True,
            "selection_split": "development",
            "basis": "meeting_exact_floor_baseline_unmeasured",
            "environment": "mixed",
        })
        # Compare parsed configuration meaning; whitespace/formatting is irrelevant.
        restored_selection = deepcopy(floor)
        restored_selection["pipeline"]["prompt_selection"] = indoor["pipeline"]["prompt_selection"]
        self.assertEqual(restored_selection, indoor)

    def test_exact_lowercase_floor_reaches_real_sam_reader_and_shared_contract(self):
        floor = read_preset("ground_surface_floor.json")
        result, processor = self.surface_observation(floor, detections=2)
        self.assertIsInstance(result, SemanticFrame)
        self.assertEqual((result.contract_id, result.schema_version), ("semantic_mapping_v1", 1))
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(processor.prompts, ["floor"])
        self.assertEqual(processor.reset_count, 1)
        self.assertEqual(len(processor.images), 1)
        np.testing.assert_array_equal(processor.images[0], self.frame.rgb)
        self.assertEqual((result.model_calls, result.query_count), ({"sam3": 1}, 1))
        self.assertEqual(len(result.queries), 1)
        query = result.queries[0]
        self.assertEqual((query.original_phrase, query.concept_id, query.role),
                         ("floor", "ground_surface", "candidate_surface"))
        self.assertEqual(query.mapping_version, "ground_surface_v1")
        self.assertEqual(len(query.instances), 2)
        for instance in query.instances:
            self.assertEqual(instance.mask.shape, self.frame.rgb.shape[:2])
            self.assertEqual(instance.mask.dtype, np.bool_)
            self.assertEqual(instance.processed_grid_id, self.frame.processed_grid_id)
            self.assertIn("uncalibrated", instance.score_meaning)
        provenance = result.adapter_provenance
        self.assertTrue(provenance["fixture"])
        self.assertFalse(provenance["timing"]["gpu_synchronized"])
        self.assertEqual(provenance["policy"]["prompt_selection"], floor["pipeline"]["prompt_selection"])
        self.assertEqual(provenance["input"]["decoded_rgb_sha256"], self.frame.decoded_rgb_sha256)

    def test_empty_floor_detection_still_records_one_candidate_surface_query(self):
        result, processor = self.surface_observation(read_preset("ground_surface_floor.json"), detections=0)
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(processor.prompts, ["floor"])
        self.assertEqual(result.query_count, 1)
        self.assertEqual(result.queries[0].instances, [])
        self.assertEqual(result.queries[0].role, "candidate_surface")

    def test_historical_surface_presets_keep_their_original_meanings(self):
        cases = [
            ("ground_surface_indoor.json", "floor", "ground_surface_indoor_v1", "indoor"),
            ("ground_surface_outdoor.json", "ground", "ground_surface_outdoor_v1", "outdoor"),
            ("ground_surface_path_legacy.json", "Path", "ground_surface_path_legacy_ablation_v1", "path_legacy"),
        ]
        indoor = read_preset("ground_surface_indoor.json")
        common = {key: value for key, value in indoor.items() if key != "pipeline"}
        for name, phrase, config_id, environment in cases:
            with self.subTest(preset=name):
                preset = read_preset(name)
                self.assertEqual({key: value for key, value in preset.items() if key != "pipeline"}, common)
                self.assertEqual(preset["pipeline"]["sam3"], indoor["pipeline"]["sam3"])
                self.assertEqual(preset["pipeline"]["config_id"], config_id)
                self.assertEqual(preset["pipeline"]["prompt_selection"], {
                    "frozen": True, "selection_split": "development",
                    "basis": "starting_proposal_unmeasured", "environment": environment,
                })
                result, processor = self.surface_observation(preset)
                self.assertEqual(result.status, "ok", result.error)
                self.assertEqual(processor.prompts, [phrase])
                self.assertEqual((result.queries[0].original_phrase, result.queries[0].concept_id,
                                  result.queries[0].role), (phrase, "ground_surface", "candidate_surface"))
                self.assertEqual(result.adapter_provenance["legacy_compatibility"]["ablation"], phrase == "Path")

    def test_rich_policy_preserves_hazards_and_explicitly_groups_surface_words(self):
        config = read_preset("fixed_hazards_rich_vocabulary.json")
        options = config["pipeline"]
        self.assertEqual(config["pipeline_id"], "fixed_hazards")
        self.assertEqual(options["vocabulary_mode"], "explicit_concepts_v1")
        self.assertEqual(options["policy_path"], RICH_POLICY_PATH)
        policy = json.loads((ROOT / options["policy_path"]).read_text(encoding="utf-8"))
        expected_phrases = HISTORICAL_HAZARDS + SURFACE_PHRASES + ADDED_HAZARDS
        self.assertEqual((policy["task_id"], policy["schema_version"], policy["policy_id"]),
                         ("hazard_prompt_v1", 1, "fixed_vocabulary_surface_hazards_v1"))
        self.assertEqual(policy["vocabulary_mode"], "explicit_concepts_v1")
        self.assertEqual(len(expected_phrases), 32)
        self.assertEqual(policy["canonical_prompts"], expected_phrases)
        expected_specs = [
            {"phrase": phrase, "concept_id": "ground_surface" if phrase in SURFACE_PHRASES else phrase,
             "role": "candidate_surface" if phrase in SURFACE_PHRASES else "hazard"}
            for phrase in expected_phrases
        ]
        self.assertEqual(policy["concepts"], expected_specs)
        self.assertEqual(set(policy["aliases"]), set(expected_phrases))
        for phrase in expected_phrases:
            self.assertIn(phrase, policy["aliases"][phrase])
        self.assertNotIn("Path", policy["canonical_prompts"])

    def test_rich_preset_keeps_the_original_pinned_sam_and_local_only_loading(self):
        rich = read_preset("fixed_hazards_rich_vocabulary.json")["pipeline"]
        historical = read_preset("fixed_hazards.json")["pipeline"]
        identity_settings = (
            "model_key", "code_repository", "code_revision", "checkpoint_repository",
            "checkpoint_revision", "checkpoint_filename", "checkpoint_sha256",
            "checkpoint_size_bytes", "device", "resolution", "confidence_threshold",
            "mask_probability_threshold", "allow_downloads", "local_files_only",
        )
        self.assertEqual({key: rich["sam_settings"][key] for key in identity_settings},
                         {key: historical["sam_settings"][key] for key in identity_settings})
        self.assertIs(rich["enable_model_loading"], True)
        self.assertIs(rich["sam_settings"]["enable_model_loading"], True)
        self.assertIs(rich["sam_settings"]["allow_downloads"], False)
        self.assertIs(rich["sam_settings"]["local_files_only"], True)
        self.assertEqual(rich["sam_settings"]["code_revision"], "2345a4ad109ac29c569da749c91d84f10dc08c40")
        self.assertEqual(rich["sam_settings"]["checkpoint_revision"], "3c879f39826c281e95690f02c7821c4de09afae7")

    def test_rich_preset_executes_all_original_words_with_declared_roles(self):
        options = read_preset("fixed_hazards_rich_vocabulary.json")["pipeline"]
        policy_path = ROOT / options["policy_path"]
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        processor = RecordingProcessor()
        segmenter = load_segmenter({
            **deepcopy(options["sam_settings"]), "fixture": True, "processor": processor,
            "policy_path": str(policy_path), "policy_id": policy["policy_id"],
        })
        adapter = create_vocabulary_adapter({
            **deepcopy(options), "fixture": True, "_segmenter": segmenter,
        })
        self.addCleanup(adapter.close)
        result = adapter.observe(self.frame)
        validate_semantic(result, self.frame)
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(processor.prompts, policy["canonical_prompts"])
        self.assertEqual(processor.reset_count, 32)
        self.assertEqual(len(processor.images), 1)
        np.testing.assert_array_equal(processor.images[0], self.frame.rgb)
        self.assertEqual(result.query_count, 32)
        self.assertEqual(len(result.queries), 32)
        self.assertEqual(sum(result.model_calls.values()), 1)
        self.assertEqual([
            {"phrase": query.original_phrase, "concept_id": query.concept_id, "role": query.role}
            for query in result.queries
        ], policy["concepts"])
        self.assertTrue(all(query.mapping_version == policy["policy_id"] for query in result.queries))
        self.assertTrue(all(len(query.instances) == 1 for query in result.queries))


if __name__ == "__main__":
    unittest.main()
