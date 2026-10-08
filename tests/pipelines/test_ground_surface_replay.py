"""CPU admission tests for the offline replay-config helper.

All scene/model metadata here are explicit test doubles. Numeric archives and
hashes are real temporary files; no checkpoint, model, CUDA or server is used.
"""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from path_mapping.runner import fixture, geometry_fingerprint
from pipeline_common.io import digest_json, file_sha256, write_json, write_npz
from pipeline_common.sequence import prepare

SPEC = importlib.util.spec_from_file_location(
    "prepare_ground_surface_replay_for_tests",
    ROOT / "docs" / "implementation" / "prepare_ground_surface_replay.py",
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class ReplayConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "geometry_only_run"
        self.cache = self.root / "shared_cache"
        self.sequence_path = self.root / "sequence" / "sequence.json"
        self.run_manifest_path = self.run / "geometry" / "manifest.json"
        self.cache_manifest_path = self.cache / "manifest.json"
        geometry, _, _ = fixture()
        sources = self.root / "inputs"
        sources.mkdir()
        for index, rgb in enumerate(geometry["images"]):
            Image.fromarray(rgb).save(sources / f"{index:03d}.png")
        self.sequence = prepare(sources, {
            "contract_id": "semantic_mapping_v1", "schema_version": 1,
            "sequence_id": "metadata_admission_test_é", "split": "development",
            "fixture": False, "max_frames": 3,
        }, self.sequence_path.parent)
        self.base = {
            "contract_id": "semantic_mapping_v1", "schema_version": 1,
            "protocol_id": "semantic_mapping_eval_v1", "pipeline_id": "geometry_only",
            "fixture": False, "mode": "quality_replay",
            "sequence": {key: self.sequence[key]
                         for key in ("sequence_id", "split", "manifest_digest")},
            "geometry": {
                "image_size": 518, "keyframe_interval": 2, "min_confidence": 1.8,
                "device": "cuda:2", "pose_revision": "frozen_pose_revision",
                "map_frame": "replay_world", "scale": {}, "up": {},
                "source_root": "/existing/frozen/lingbot",
                "checkpoint": "/existing/frozen/lingbot-map.pt",
                "checkpoint_sha256": "a" * 64,
            },
            "fusion": {"voxel_size": 0.11, "origin": [-3, 4, 5],
                       "semantic_threshold": 0.73, "negative_evidence": True},
            "robot": {"version": "admission_test_profile", "height": 0.47,
                      "footprint_radius": 0.13, "clearance": 0.06},
            "planning": {"shape": [4, 7], "resolution": 0.3, "support_height": 0.4},
            "goals": [{"request_id": "frozen_goal", "frame": "replay_world",
                       "start": [1, 2, 3], "goal": [4, 5, 6]}],
            "semantic_keyframe_ids": [self.sequence["frames"][i]["frame_id"] for i in (0, 2)],
            "runtime": {"cadence_frames": 2, "pending_capacity": 1,
                        "playback_rate": 0.8, "expiry_ns": None},
            "evaluation_grid": {"shape": [4, 7], "origin": [-3, 4, 0],
                                "resolution": 0.3, "map_frame": "replay_world"},
            "evaluation": {"semantic_evidence_threshold": 0.77,
                           "reference_protocol_id": "independent_reference_protocol"},
            "hardware_budget": {"device": "cuda:2", "max_visible_gpus": 1,
                                "measured": False},
            "pipeline_config": {"pipeline_id": "geometry_only"},
            "pipelines": {"ground_surface": {"prompt": "stale_override"},
                          "fixed_hazards": {"unchanged_other_adapter": True}},
            "config_digest": "old_geometry_only_digest",
        }
        self.geometry_archive = self.cache / "geometry.npz"
        write_npz(self.geometry_archive, **geometry)
        input_identity = {
            "contract_id": "semantic_mapping_v1", "schema_version": 1,
            "sequence_digest": self.sequence["manifest_digest"],
            "settings": self.geometry_identity(self.base["geometry"], False),
            "transforms": [{"recipe": "test_double_processed_grid"}],
        }
        manifest = {
            "contract_id": "semantic_mapping_v1", "schema_version": 1,
            "status": "complete", "input_identity": input_identity,
            "input_fingerprint": digest_json(input_identity),
            "geometry_fingerprint": geometry_fingerprint(geometry),
            "processed_grid_id": "frozen_grid_identity",
            "archive_sha256": file_sha256(self.geometry_archive),
            "map_frame": "replay_world", "units": "reconstruction_units",
            "up": None, "scale": {}, "pose_revision": "frozen_pose_revision",
            "model": {"fixture": False, "test_double": True},
        }
        self.base["geometry_identity"] = {key: manifest[key] for key in (
            "geometry_fingerprint", "input_fingerprint", "processed_grid_id",
            "units", "up", "scale", "pose_revision",
        )}
        write_json(self.cache_manifest_path, manifest)
        write_json(self.run_manifest_path, {**manifest, "cache_path": str(self.cache)})
        write_json(self.run / "config.resolved.json", self.base)
        write_json(self.run / "run.json", {
            "contract_id": "semantic_mapping_v1", "schema_version": 1,
            "pipeline_id": "geometry_only", "fixture": False, "status": "complete",
        })
        versions_patch = patch.object(helper, "_versions", return_value={
            "python": "3.12.mock", "verification_scope": "CPU test metadata",
        })
        self.mock_versions = versions_patch.start()
        self.addCleanup(versions_patch.stop)
        assets_patch = patch.object(helper, "_verify_sam_assets", side_effect=self.verify_assets)
        self.mock_assets = assets_patch.start()
        self.addCleanup(assets_patch.stop)
        identity_patch = patch(
            "pipeline_common.geometry.geometry_settings_identity", side_effect=self.geometry_identity,
        )
        self.mock_geometry_identity = identity_patch.start()
        self.addCleanup(identity_patch.stop)

    @staticmethod
    def geometry_identity(settings, fixture_mode):
        return {**deepcopy(settings), "fixture": fixture_mode,
                "actual_code": {"source_revision": "synthetic_cpu_test_source",
                                "upstream_python_code_sha256": "b" * 64}}

    def verify_assets(self, settings):
        # Mirror the verifier's resolved-file mutation without loading libraries.
        self.assertFalse(settings["allow_downloads"])
        self.assertTrue(settings["local_files_only"])
        self.asset_settings = deepcopy(settings)
        resolved = self.root / "mock_verified_assets" / "sam3.pt"
        settings.update(cache_root=str(resolved.parent), checkpoint_path=resolved.name)
        return {"checkpoint": str(resolved), "downloads": False, "model_loaded": False,
                "checkpoint_verification": {"status": "verified", "test_double": True},
                "code_verification": {"status": "verified", "test_double": True}}

    def build(self, environment="outdoor", **kwargs):
        return helper.build_replay_config(self.run, self.sequence_path, environment, **kwargs)

    def rewrite(self, path, mutate):
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        write_json(path, value)

    def assert_rejected_before_assets(self, expected="", **kwargs):
        with self.assertRaisesRegex((ValueError, FileNotFoundError), expected):
            self.build(**kwargs)
        self.mock_versions.assert_not_called()
        self.mock_assets.assert_not_called()

    def test_preserves_frozen_common_experiment_and_selects_exact_environment_phrase(self):
        original_files = {path: path.read_bytes() for path in (
            self.run / "run.json", self.run / "config.resolved.json", self.run_manifest_path,
            self.cache_manifest_path, self.geometry_archive, self.sequence_path,
        )}
        for environment, phrase in (("indoor", "floor"), ("outdoor", "ground")):
            with self.subTest(environment=environment):
                prepared, report = self.build(environment)
                for key in helper.PRESERVED_FIELDS:
                    self.assertEqual(prepared[key], self.base[key], key)
                self.assertEqual(prepared["evaluation"], self.base["evaluation"])
                self.assertEqual(prepared["pipeline_id"], "ground_surface")
                self.assertFalse(prepared["fixture"])
                self.assertNotIn("config_digest", prepared)
                self.assertEqual(prepared["pipeline"]["prompt"], phrase)
                self.assertEqual(prepared["pipeline_config"], prepared["pipeline"])
                self.assertEqual(prepared["pipelines"]["ground_surface"], prepared["pipeline"])
                self.assertEqual(prepared["pipelines"]["fixed_hazards"],
                                 self.base["pipelines"]["fixed_hazards"])
                self.assertEqual(prepared["pipeline_config"]["sam3"]["device"], "cuda:2")
                self.assertIsNone(self.asset_settings["checkpoint_path"])
                self.assertEqual(report["geometry_cache"], str(self.cache.resolve()))
                self.assertEqual(report["status"], "assets_verified")
                self.assertIsNone(report["gpu_ready"])
                self.assertFalse(report["sam_assets"]["model_loaded"])
                for key, path in (("geometry_run_sha256", self.run / "run.json"),
                                  ("geometry_config_sha256", self.run / "config.resolved.json"),
                                  ("geometry_manifest_sha256", self.run_manifest_path),
                                  ("sequence_manifest_sha256", self.sequence_path)):
                    self.assertEqual(report[key], file_sha256(path))
        for path, content in original_files.items():
            self.assertEqual(path.read_bytes(), content, str(path))

    def test_rejects_fixture_and_missing_explicit_real_markers(self):
        for path in (self.run / "run.json", self.run / "config.resolved.json", self.sequence_path):
            original = path.read_bytes()
            for marker in (True, None, 0):
                with self.subTest(path=path.name, marker=marker):
                    path.write_bytes(original)
                    self.rewrite(path, lambda record: record.update(fixture=marker))
                    self.assert_rejected_before_assets("nonfixture")
            path.write_bytes(original)

    def test_rejects_incomplete_run_run_geometry_manifest_and_cache(self):
        for path in (self.run / "run.json", self.run_manifest_path, self.cache_manifest_path):
            original = path.read_bytes()
            with self.subTest(path=str(path)):
                self.rewrite(path, lambda record: record.update(status="running"))
                self.assert_rejected_before_assets("complete")
            path.write_bytes(original)

    def test_rejects_wrong_pipeline_and_non_quality_replay(self):
        self.rewrite(self.run / "run.json", lambda record: record.update(pipeline_id="fixed_hazards"))
        self.assert_rejected_before_assets("geometry_only")
        self.rewrite(self.run / "run.json", lambda record: record.update(pipeline_id="geometry_only"))
        self.rewrite(self.run / "config.resolved.json", lambda record: record.update(mode="paced_runtime"))
        self.assert_rejected_before_assets("quality_replay")

    def test_rejects_cached_contract_and_schema_mismatch(self):
        original = self.cache_manifest_path.read_bytes()
        for update in ({"contract_id": "another_contract"}, {"schema_version": 2},
                       {"schema_version": True}):
            with self.subTest(update=update):
                self.cache_manifest_path.write_bytes(original)
                self.rewrite(self.cache_manifest_path, lambda record: record.update(update))
                self.assert_rejected_before_assets("Cached geometry contract/schema")

    def test_rejects_archive_bytes_tampering_before_sam_verification(self):
        with self.geometry_archive.open("ab") as stream:
            stream.write(b"tampering that leaves an otherwise readable zip")
        self.assert_rejected_before_assets("archive bytes")

    def test_rejects_cached_input_identity_tampering_with_stale_matching_fingerprint(self):
        # The run/cache still agree on the old fingerprint, so merely joining
        # those values cannot detect altered preprocessing/crop provenance.
        self.rewrite(self.cache_manifest_path,
                     lambda record: record["input_identity"]["transforms"].append({"crop_xywh": [1, 2, 3, 4]}))
        self.assert_rejected_before_assets("fingerprint|input identity")

    def test_rejects_run_cache_grid_pose_or_archive_identity_disagreement(self):
        original = self.cache_manifest_path.read_bytes()
        for key in ("processed_grid_id", "pose_revision", "archive_sha256"):
            with self.subTest(key=key):
                self.cache_manifest_path.write_bytes(original)
                self.rewrite(self.cache_manifest_path, lambda record: record.update({key: "changed"}))
                self.assert_rejected_before_assets("cache/run")

    def test_rejects_changed_frozen_geometry_settings_and_fixture_cache_identity(self):
        original = self.run / "config.resolved.json"
        saved = original.read_bytes()
        self.rewrite(original, lambda record: record["geometry"].update(min_confidence=0.1))
        self.assert_rejected_before_assets("geometry settings")
        original.write_bytes(saved)
        self.rewrite(self.cache_manifest_path,
                     lambda record: record["input_identity"]["settings"].update(fixture=True))
        self.assert_rejected_before_assets("geometry settings|fingerprint")

    def test_rejects_resolved_geometry_identity_mismatch(self):
        self.rewrite(self.run / "config.resolved.json",
                     lambda record: record["geometry_identity"].update(processed_grid_id="different"))
        self.assert_rejected_before_assets("Resolved geometry processed_grid_id")

    def test_rejects_sequence_manifest_tampering_and_other_cache_sequence(self):
        original = self.sequence_path.read_bytes()
        self.rewrite(self.sequence_path, lambda record: record.update(split="test"))
        self.assert_rejected_before_assets("Sequence manifest digest")
        self.sequence_path.write_bytes(original)
        self.rewrite(self.cache_manifest_path,
                     lambda record: record["input_identity"].update(sequence_digest="different"))
        self.assert_rejected_before_assets("another sequence|fingerprint")

    def test_local_checkpoint_override_requires_existing_file_and_remains_local_only(self):
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint is absent"):
            self.build(sam_checkpoint=self.root / "absent.pt")
        self.mock_assets.assert_not_called()
        checkpoint = self.root / "mock_local_checkpoint.pt"
        checkpoint.write_bytes(b"CPU test checkpoint identity placeholder; verifier is mocked")
        prepared, report = self.build(sam_checkpoint=checkpoint, sam_source=self.root / "pinned_source")
        self.assertEqual(self.asset_settings["cache_root"], str(checkpoint.parent.resolve()))
        self.assertEqual(self.asset_settings["checkpoint_path"], checkpoint.name)
        self.assertEqual(self.asset_settings["code_source_root"], str((self.root / "pinned_source").resolve()))
        self.assertTrue(prepared["pipeline_config"]["sam3"]["local_files_only"])
        self.assertFalse(report["model_settings"]["allow_downloads"])

    def cli(self, output):
        return helper.main([
            "--geometry-run", str(self.run), "--sequence", str(self.sequence_path),
            "--environment", "indoor", "--output", str(output),
        ])

    def test_cli_writes_inspectable_config_and_preflight_without_mutating_inputs(self):
        output = self.root / "new_output" / "indoor_replay.json"
        with redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(self.cli(output), 0)
        report_path = output.with_suffix(".preflight.json")
        prepared = json.loads(output.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(prepared["pipeline_config"]["prompt"], "floor")
        self.assertEqual(prepared["geometry"], self.base["geometry"])
        self.assertEqual(report["geometry_config_sha256"], file_sha256(self.run / "config.resolved.json"))
        self.assertEqual(json.loads(stdout.getvalue())["config"], str(output.resolve()))

    def test_cli_refuses_existing_config_or_preflight_without_touching_them(self):
        output = self.root / "replay.json"
        report_path = output.with_suffix(".preflight.json")
        for existing in (output, report_path):
            with self.subTest(existing=existing.name):
                existing.write_bytes(b"existing user artifact")
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                    self.cli(output)
                self.assertEqual(stopped.exception.code, 2)
                self.assertEqual(existing.read_bytes(), b"existing user artifact")
                other = report_path if existing == output else output
                self.assertFalse(other.exists())
                existing.unlink()
        self.mock_versions.assert_not_called()
        self.mock_assets.assert_not_called()

    def test_cli_uses_distinct_artifact_paths_even_with_preflight_output_suffix(self):
        output = self.root / "replay.preflight.json"
        report_path = output.with_suffix(".preflight.json")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.cli(output), 0)
        self.assertNotEqual(output, report_path)
        self.assertEqual(json.loads(output.read_text())["pipeline_id"], "ground_surface")
        self.assertEqual(json.loads(report_path.read_text())["status"], "assets_verified")

    def test_cli_blocked_cache_writes_no_success_artifact(self):
        output = self.root / "blocked_config.json"
        self.rewrite(self.cache_manifest_path, lambda record: record.update(status="failed"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            self.cli(output)
        self.assertEqual(stopped.exception.code, 2)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".preflight.json").exists())
        self.mock_versions.assert_not_called()
        self.mock_assets.assert_not_called()


if __name__ == "__main__":
    unittest.main()
