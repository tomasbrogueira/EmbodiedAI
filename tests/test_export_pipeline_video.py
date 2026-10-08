"""Read-only video-export artifact and chronology regressions; no model runs."""
from __future__ import annotations

import copy
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np
from PIL import Image

from export_pipeline_video import (
    cumulative_states,
    export_movies,
    load_export_inputs,
    load_npz_checked,
    project_mask_to_source,
    sampled_timeline,
    voxel_faces,
)


PROJECT = Path(__file__).resolve().parents[1]


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _save_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class _FixtureProcessor:
    """Explicit CPU processor behind the production SAM reader and input bridge."""
    def __init__(self):
        self.images = []
        self.prompts = []

    def set_image(self, image):
        self.images.append(np.array(image))
        return {}

    def reset_all_prompts(self, state):
        state.clear()

    def set_text_prompt(self, *, state, prompt):
        self.prompts.append(prompt)
        height, width = self.images[-1].shape[:2]
        masks = np.ones((2, 1, height, width), dtype=bool)
        masks[1, :, :, :width // 2] = False
        return {"masks": masks, "scores": np.array([.8, .65])}


class SavedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pipeline_common.fixture import create_fixture
        from pipeline_common.runtime import run
        from pipelines.ground_surface import create_adapter
        from traversability_hazard_segmentation.sam3_adapter import Sam3Segmenter

        cls.reference_temp = tempfile.TemporaryDirectory(prefix="pipeline-video-fixture-")
        cls.addClassCleanup(cls.reference_temp.cleanup)
        reference_root = Path(cls.reference_temp.name)
        cls.common_fixture = create_fixture(reference_root / "analytic_fixture")
        cls.adapter_fixture = reference_root / "adapter_run"
        config = json.loads((cls.common_fixture / "run.json").read_text(encoding="utf-8"))
        config["pipeline_config"] = {"prompt": "ground"}
        config_path = reference_root / "ground_surface.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        processor = _FixtureProcessor()
        segmenter = Sam3Segmenter(
            {"fixture": True, "fixture_id": "export_pipeline_video_processor_v1"},
            processor=processor,
        )
        cls.addClassCleanup(segmenter.close)

        def create_ground_fixture(pipeline, settings, sequence):
            if (pipeline != "ground_surface" or settings.get("fixture") is not True
                    or sequence.get("fixture") is not True):
                raise AssertionError("Video-export fixture requires explicit ground-surface fixture identity")
            return create_adapter({**settings, "_segmenter": segmenter})

        with patch("pipeline_common.fixture_adapters.create_fixture_adapter",
                   side_effect=create_ground_fixture):
            run("ground_surface", cls.common_fixture / "sequence/sequence.json",
                config_path, cls.adapter_fixture, fixture=True,
                cache=cls.common_fixture / "geometry_cache")
        if processor.prompts != ["ground"] * 3:
            raise AssertionError("Fixture must execute one exact ground query per analytic frame")
        with np.load(cls.common_fixture / "geometry_cache/geometry.npz", allow_pickle=False) as geometry:
            np.testing.assert_array_equal(np.stack(processor.images), geometry["images"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.sequence = self.root / "sequence/sequence.json"
        self.cache = self.root / "cache"
        shutil.copytree(self.adapter_fixture, self.run)
        shutil.copytree(self.common_fixture / "sequence", self.sequence.parent)
        shutil.copytree(self.common_fixture / "geometry_cache", self.cache)

    def load(self):
        return load_export_inputs(self.run, self.sequence, cache_dir=self.cache)

    def semantic_rows(self):
        return _rows(self.run / "semantics/frames.jsonl")

    def replace_semantics(self, rows):
        _save_rows(self.run / "semantics/frames.jsonl", rows)

    def first_mask(self):
        return self.run / self.semantic_rows()[0]["queries"][0]["instances"][0]["mask_path"]

    def test_actual_adapter_fixture_retains_exact_rgb_frame_and_query_identity(self):
        data = self.load()
        self.assertEqual([f["frame_id"] for f in data["frames"]],
                         [f"frame_{i:06d}" for i in range(3)])
        self.assertEqual([f["timestamp_ns"] for f in data["frames"]], [0, 100000000, 200000000])
        with np.load(self.cache / "geometry.npz", allow_pickle=False) as geometry:
            for index, frame in enumerate(data["frames"]):
                np.testing.assert_array_equal(frame["rgb"], geometry["images"][index])
                self.assertEqual(frame["rgb"].dtype, np.uint8)
                self.assertEqual(frame["semantic"]["status"], "ok")
                query = frame["semantic"]["queries"][0]
                self.assertEqual((query["original_phrase"], query["role"], query["status"]),
                                 ("ground", "candidate_surface", "ok"))
                self.assertEqual(len(query["instances"]), 2)

    def test_source_png_hash_change_is_refused(self):
        source = self.sequence.parent / "images/000000.png"
        with Image.open(source) as image:
            pixels = np.array(image.convert("RGB"))
        pixels[0, 0, 0] ^= 1
        Image.fromarray(pixels).save(source)
        with self.assertRaises(ValueError):
            self.load()

    def test_processed_png_bytes_change_is_refused_even_with_same_decoded_rgb(self):
        path = self.cache / "processed_frames/000000.png"
        path.write_bytes(path.read_bytes() + b"appended-unrecorded-bytes")
        with self.assertRaises(ValueError):
            self.load()

    def test_cache_archive_hash_change_is_refused(self):
        path = self.cache / "geometry.npz"
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaises(ValueError):
            self.load()

    def test_mask_archive_hash_change_is_refused(self):
        path = self.first_mask()
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaises(ValueError):
            self.load()

    def test_wrong_shape_or_nonboolean_masks_are_refused_even_when_rehashed(self):
        original = self.semantic_rows()
        mask_path = self.first_mask()
        for mask in (np.zeros((3, 3), dtype=bool), np.ones((12, 16), dtype=np.uint8)):
            np.savez_compressed(mask_path, mask=mask)
            rows = copy.deepcopy(original)
            rows[0]["queries"][0]["instances"][0]["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
            self.replace_semantics(rows)
            with self.subTest(shape=mask.shape, dtype=str(mask.dtype)), self.assertRaises(ValueError):
                self.load()

    def test_missing_mask_is_an_explicit_failure(self):
        self.first_mask().unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.load()

    def test_unsafe_mask_path_is_refused_before_asset_access(self):
        for unsafe in ("../outside.npz", str(self.root / "outside.npz")):
            rows = self.semantic_rows()
            rows[0]["queries"][0]["instances"][0]["mask_path"] = unsafe
            self.replace_semantics(rows)
            with self.subTest(path=unsafe), self.assertRaises(ValueError):
                self.load()

    def test_mask_grid_and_semantic_geometry_timestamp_join_are_verified(self):
        original = self.semantic_rows()
        for field, value in (("processed_grid_id", "wrong-grid"),
                             ("geometry_fingerprint", "wrong-geometry"),
                             ("timestamp_ns", 12345), ("frame_id", "unknown-frame")):
            rows = copy.deepcopy(original)
            rows[0][field] = value
            self.replace_semantics(rows)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load()

    def test_successful_empty_and_failed_query_statuses_are_preserved(self):
        rows = self.semantic_rows()
        rows[0]["queries"][0]["instances"] = []
        rows[1]["status"] = "error"
        rows[1]["error"] = {"code": "sam_query_error", "message": "synthetic test failure"}
        rows[1]["queries"][0]["status"] = "error"
        rows[1]["queries"][0]["error"] = rows[1]["error"]
        self.replace_semantics(rows)
        journal_path = self.run / "map/contributions.jsonl"
        journal = _rows(journal_path)
        changed = []
        for record in journal:
            if record["frame_id"] == "frame_000000" and record["kind"] == "semantic_query":
                for field in ("observation_ids", "mask_sha256", "instance_scores", "score_meanings"):
                    record[field] = []
            if record["frame_id"] == "frame_000000" and record["kind"] == "semantic":
                record["observation_ids"] = []
                record["positive_weight"] = [0.0] * len(record["positive_weight"])
            if record["frame_id"] == "frame_000001" and record["kind"] == "semantic_query":
                record.update(status="error", frame_status="error", fused=False, error=rows[1]["error"])
            if record["frame_id"] == "frame_000001" and record["kind"] == "semantic":
                continue
            changed.append(record)
        _save_rows(journal_path, changed)
        dispositions = _rows(self.run / "frames.jsonl")
        dispositions[1].update(status="error", semantic_status="error", error=rows[1]["error"])
        _save_rows(self.run / "frames.jsonl", dispositions)
        evidence_path = self.run / "map/semantic_evidence.npz"
        with np.load(evidence_path, allow_pickle=False) as saved:
            evidence = {name: saved[name] for name in saved.files}
        evidence["positive_weight"] /= 3
        evidence["observed_weight"] *= 2 / 3
        evidence["frame_support"] -= 1
        evidence["evidence_score"] = evidence["positive_weight"] / evidence["observed_weight"]
        np.savez_compressed(evidence_path, **evidence)
        data = self.load()
        self.assertEqual(data["frames"][0]["semantic"]["queries"][0]["instances"], [])
        self.assertEqual(data["frames"][0]["semantic"]["queries"][0]["status"], "ok")
        self.assertEqual(data["frames"][1]["semantic"]["status"], "error")
        self.assertEqual(data["frames"][1]["semantic"]["queries"][0]["status"], "error")
        states = cumulative_states(data)
        self.assertEqual(len(states[0]["positive_indices"]), 0)
        self.assertEqual(len(states[1]["positive_indices"]), 0)
        self.assertGreater(len(states[2]["positive_indices"]), 0)

    def test_partial_frame_keeps_successful_query_fused(self):
        rows = self.semantic_rows()
        rows[0]["status"] = "partial"
        self.replace_semantics(rows)
        journal_path = self.run / "map/contributions.jsonl"
        journal = _rows(journal_path)
        for record in journal:
            if record["frame_id"] == "frame_000000" and record["kind"] == "semantic_query":
                record["frame_status"] = "partial"
        _save_rows(journal_path, journal)
        data = self.load()
        self.assertEqual(data["frames"][0]["semantic"]["status"], "partial")
        self.assertGreater(len(cumulative_states(data)[0]["positive_indices"]), 0)

    def test_native_saved_path_only_appears_on_final_cumulative_map(self):
        data = self.load()
        states = cumulative_states(data)
        self.assertEqual(states[0]["native_paths"], [])
        self.assertEqual(states[1]["native_paths"], [])
        self.assertEqual(len(states[2]["native_paths"]), 1)
        expected = _rows(self.run / "planning/plans.jsonl")[0]["path"]
        np.testing.assert_array_equal(states[2]["native_paths"][0], expected)

    def research_plan(self):
        geometry = json.loads((self.run / "geometry/manifest.json").read_text())
        return {"schema_version": 1, "artifact_kind": "research_illustration_plan",
                "research_illustration": True, "safety_validated": False,
                "frame_scope": "final_cumulative_map_posthoc", "status": "ok",
                "source": {key: geometry[key] for key in ("geometry_fingerprint",
                            "input_fingerprint", "processed_grid_id", "archive_sha256",
                            "map_frame", "units")},
                "path_points": [[-.75, -.25, 0], [.75, -.25, 0]]}

    def test_research_path_is_separate_and_final_only(self):
        plan = self.research_plan()
        path = self.root / "research_plan.json"
        path.write_text(json.dumps(plan))
        data = load_export_inputs(self.run, self.sequence, cache_dir=self.cache,
                                  research_plan_path=path)
        states = cumulative_states(data)
        self.assertEqual(len(states[0]["research_path"]), 0)
        self.assertEqual(len(states[1]["research_path"]), 0)
        np.testing.assert_array_equal(states[2]["research_path"], plan["path_points"])
        self.assertEqual(len(states[2]["native_paths"]), 1)
        self.assertFalse(data["research_plan"]["safety_validated"])

    def test_research_source_identity_and_safety_scope_are_verified(self):
        original = self.research_plan()
        path = self.root / "research_plan.json"
        cases = []
        for key in original["source"]:
            altered = copy.deepcopy(original)
            altered["source"][key] = "wrong-identity"
            cases.append((key, altered))
        for key, value in (("safety_validated", True), ("research_illustration", False),
                           ("frame_scope", "online")):
            altered = copy.deepcopy(original)
            altered[key] = value
            cases.append((key, altered))
        for field, plan in cases:
            path.write_text(json.dumps(plan))
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_export_inputs(self.run, self.sequence, cache_dir=self.cache,
                                   research_plan_path=path)

    def test_unavailable_native_planning_does_not_draw_a_saved_path(self):
        path = self.run / "planning/manifest.json"
        planning = json.loads(path.read_text())
        planning.update(availability="blocked_inputs", reason="unknown robot, scale and up")
        path.write_text(json.dumps(planning))
        self.assertTrue(all(state["native_paths"] == [] for state in cumulative_states(self.load())))

    def test_frames_only_export_keeps_true_counts_when_render_is_capped(self):
        data = self.load()
        input_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in self.root.rglob("*") if path.is_file()}
        output = self.root / "export"
        manifest = export_movies(data, output, panel_size=400, max_voxels=1,
                                 frames_only=True)
        self.assertEqual(manifest["status"], "complete_frames_only")
        self.assertEqual(manifest["sampled_frame_count"], 3)
        self.assertEqual(manifest["coverage"][-1]["cumulative_voxels"], 88)
        config = json.loads((output / "export_config.json").read_text())
        self.assertEqual(config["selected_voxels"], 1)
        self.assertEqual(len(list((output / "combined_frames").glob("*.png"))), 3)
        with Image.open(output / "combined_frames/000000.png") as preview:
            self.assertEqual(preview.size, (800, 400))
        self.assertEqual({path: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in input_hashes}, input_hashes)

    def test_export_refuses_saved_inputs_and_existing_output(self):
        data = self.load()
        for output in (self.run / "export", self.cache / "export",
                       self.sequence.parent / "export", self.run):
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                export_movies(data, output, panel_size=400, frames_only=True)

    def test_cumulative_journal_uses_only_current_and_prior_frames(self):
        data = self.load()
        indices = data["final_voxel_indices"][:3].tolist()
        data["final_voxel_indices"] = np.asarray(indices, dtype=np.int64)
        revised = []
        for index, frame in enumerate(data["frames"]):
            revised.extend([
                {"kind": "geometry", "status": "ok", "frame_id": frame["frame_id"],
                 "voxel_indices": [indices[index]], "timestamp_ns": frame["timestamp_ns"]},
                {"kind": "semantic", "frame_id": frame["frame_id"],
                 "voxel_indices": [indices[index]], "positive_weight": [1.0],
                 "observed_weight": [2.0], "role": "candidate_surface",
                 "concept_id": "ground_surface", "timestamp_ns": frame["timestamp_ns"]},
            ])
        data["contributions"] = list(reversed(revised))  # Saved row order cannot leak later frames.
        states = cumulative_states(data)
        for index, state in enumerate(states):
            expected = {tuple(item) for item in indices[:index + 1]}
            self.assertEqual({tuple(item) for item in state["voxel_indices"]}, expected)
            self.assertEqual({tuple(item) for item in state["positive_indices"]}, expected)
            self.assertTrue(all(score == .5 for score in state["scores"].values()))

    def test_camera_trajectory_is_current_and_past_and_separate_from_paths(self):
        data = self.load()
        states = cumulative_states(data)
        for index, state in enumerate(states):
            np.testing.assert_array_equal(state["camera_trajectory"],
                                          data["camera_trajectory"][:index + 1])
            self.assertEqual(len(state["camera_trajectory"]), index + 1)
        self.assertEqual(states[0]["native_paths"], [])
        self.assertEqual(len(states[0]["research_path"]), 0)

    def test_camera_trajectory_must_match_saved_geometry_and_timestamps(self):
        path = self.run / "geometry/camera_trajectory.npz"
        with np.load(path, allow_pickle=False) as saved:
            original = {name: saved[name] for name in saved.files}
        for field in ("world_to_camera", "timestamp_ns"):
            arrays = {name: array.copy() for name, array in original.items()}
            if field == "world_to_camera":
                arrays[field][0, 0, 3] += .123
            else:
                arrays[field][0] += 123
            np.savez_compressed(path, **arrays)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load()

    def test_semantic_expiry_uses_capture_time_and_preserves_geometry(self):
        data = self.load()
        data["map"]["semantic_expiry_ns"] = 50000000
        data["contributions"] = [row for row in data["contributions"]
                                 if row["kind"] != "semantic" or row["frame_id"] == "frame_000000"]
        states = cumulative_states(data)
        self.assertGreater(len(states[0]["positive_indices"]), 0)
        self.assertEqual(len(states[1]["positive_indices"]), 0)
        self.assertEqual(len(states[2]["positive_indices"]), 0)
        self.assertEqual(len(states[2]["voxel_indices"]), 88)

    def test_semantic_expiry_refuses_unknown_capture_time_as_current_evidence(self):
        data = self.load()
        data["map"]["semantic_expiry_ns"] = 50000000
        data["frames"][0]["timestamp_ns"] = None
        states = cumulative_states(data)
        self.assertEqual(len(states[0]["positive_indices"]), 0)

    def test_loading_does_not_write_to_saved_inputs(self):
        def hashes():
            return {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in self.root.rglob("*") if path.is_file()}
        before = hashes()
        self.load()
        self.assertEqual(hashes(), before)

    def test_loading_is_independent_of_models_and_producer_modules(self):
        code = """
import importlib.abc, sys
class RefuseModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ('torch', 'sam3', 'lingbot', 'pipeline_common.runtime',
                     'pipeline_common.geometry', 'pipeline_common.fusion',
                     'pipeline_common.planning', 'pipelines', 'path_mapping')
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise AssertionError('Producer/model import: ' + fullname)
sys.meta_path.insert(0, RefuseModels())
import export_pipeline_video as exporter
data = exporter.load_export_inputs(sys.argv[1], sys.argv[2], cache_dir=sys.argv[3])
assert len(data['frames']) == 3
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT / "src")
        process = subprocess.run([sys.executable, "-c", code, str(self.run),
                                  str(self.sequence), str(self.cache)],
                                 env=environment, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


class TimelineTests(unittest.TestCase):
    def test_assumed_ground_rotates_display_only_and_preserves_native_distances(self):
        from export_pipeline_video import display_view_matrix, _view_matrix
        up = np.array([.2, -1., .1]); up /= np.linalg.norm(up)
        view = display_view_matrix({'assumed_up_vector': up.tolist()})
        np.testing.assert_allclose(view.T @ view, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(up @ view, np.array([0., 0., 1.]) @ _view_matrix(), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(view), 1.)
        np.testing.assert_array_equal(display_view_matrix(None), _view_matrix())
        with self.assertRaises(ValueError):
            display_view_matrix({'assumed_up_vector': [0, 0, 0]})

    def test_video_presentation_clock_is_not_labeled_capture_time(self):
        from export_pipeline_video import _time_label
        provenance = {'clock': 'video_presentation_timeline', 'kind': 'decoder_reported_pts'}
        frame = {'timestamp_ns': 500000000, 'source': {'timestamp_provenance': provenance}}
        self.assertEqual(_time_label(frame, 0), 'video time 0.500s')
        self.assertEqual(sampled_timeline([frame])['timestamp_provenance'], [provenance])

    def test_irregular_sample_capture_gaps_drive_playback(self):
        frames = [{"timestamp_ns": t} for t in (0, 100000000, 600000000)]
        timeline = sampled_timeline(frames, video_fps=20, end_hold_seconds=2)
        np.testing.assert_allclose(timeline["durations_seconds"], [.1, .5, 2])
        self.assertEqual(list(timeline["repeat_counts"]), [2, 10, 40])
        self.assertTrue(timeline["playback_mode"])

    def test_unknown_capture_times_use_explicit_fallback(self):
        timeline = sampled_timeline([{"timestamp_ns": None}] * 3,
                                    video_fps=20, unknown_frame_seconds=.5,
                                    end_hold_seconds=2)
        np.testing.assert_allclose(timeline["durations_seconds"], [.5, .5, 2])
        self.assertEqual(list(timeline["repeat_counts"]), [10, 10, 40])
        self.assertTrue(timeline["playback_mode"])

    def test_invalid_playback_parameters_are_refused(self):
        frames = [{"timestamp_ns": 0}]
        for kwargs in ({"video_fps": 0}, {"video_fps": -1},
                       {"unknown_frame_seconds": 0}, {"end_hold_seconds": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sampled_timeline(frames, **kwargs)

    def test_duplicate_or_decreasing_known_capture_times_are_refused(self):
        for times in ((0, 0), (100000000, 0)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                sampled_timeline([{"timestamp_ns": value} for value in times])

    def test_mixed_capture_clock_is_explicitly_paced_by_frame_index(self):
        timeline = sampled_timeline([{"timestamp_ns": 0}, {"timestamp_ns": None},
                                     {"timestamp_ns": 600000000}])
        self.assertEqual(timeline["playback_mode"], "unknown_source_times_frame_index_pacing")
        np.testing.assert_allclose(timeline["durations_seconds"], [.5, .5, 2])


class ArchiveSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "artifact.npz"

    def test_small_numeric_archive_is_loaded_without_pickle(self):
        array = np.array([[-2, 0, 1], [3, 4, 5]], dtype=np.int64)
        np.savez_compressed(self.path, voxel_indices=array)
        loaded = load_npz_checked(self.path)
        np.testing.assert_array_equal(loaded["voxel_indices"], array)

    def test_object_array_is_refused(self):
        np.savez(self.path, data=np.array([{"unsafe": "pickle"}], dtype=object))
        with self.assertRaises(ValueError):
            load_npz_checked(self.path)


    def test_uncompressed_archive_size_limit_is_checked(self):
        np.savez_compressed(self.path, data=np.zeros(1024, dtype=np.float64))
        with self.assertRaises(ValueError):
            load_npz_checked(self.path, max_bytes=1024)

    def test_forged_huge_npy_header_is_rejected_before_allocation(self):
        buffer = BytesIO()
        np.lib.format.write_array_header_1_0(buffer, {
            "descr": "<f8", "fortran_order": False, "shape": (2 ** 40, 3),
        })
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr("data.npy", buffer.getvalue())
        with self.assertRaises(ValueError):
            load_npz_checked(self.path, max_bytes=1024 * 1024)

    def test_duplicate_npy_member_is_refused(self):
        buffer = BytesIO()
        np.save(buffer, np.array([1], dtype=np.int64), allow_pickle=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.path, "w") as archive:
                archive.writestr("data.npy", buffer.getvalue())
                archive.writestr("data.npy", buffer.getvalue())
        with self.assertRaises(ValueError):
            load_npz_checked(self.path)

    def test_truncated_numeric_payload_is_refused(self):
        buffer = BytesIO()
        np.lib.format.write_array_header_1_0(buffer, {
            "descr": "<f8", "fortran_order": False, "shape": (4,),
        })
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr("data.npy", buffer.getvalue() + bytes(8))
        with self.assertRaises(ValueError):
            load_npz_checked(self.path)


class CuboidTests(unittest.TestCase):
    def test_true_faces_use_grid_boundaries_and_preserve_negative_indices(self):
        indices = np.array([[-2, 0, 1], [3, -4, 0]], dtype=np.int64)
        origin = np.array([10.0, -5.0, .5])
        size = .25
        faces = voxel_faces(indices, origin, size)
        self.assertEqual(faces.shape, (2, 6, 4, 3))
        lower = origin + indices * size
        np.testing.assert_allclose(faces.min(axis=(1, 2)), lower)
        np.testing.assert_allclose(faces.max(axis=(1, 2)), lower + size)
        for cube in faces:
            self.assertEqual(len(np.unique(cube.reshape(-1, 3), axis=0)), 8)
            for face in cube:
                self.assertEqual(np.count_nonzero(np.ptp(face, axis=0) == 0), 1)
                self.assertEqual(len(np.unique(face, axis=0)), 4)

    def test_adjacent_cubes_share_the_exact_voxel_boundary(self):
        faces = voxel_faces(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
                            np.zeros(3), .25)
        self.assertEqual(faces[0, :, :, 0].max(), faces[1, :, :, 0].min())


class SourceMaskProjectionTests(unittest.TestCase):
    def test_identity_keeps_exact_source_pixels(self):
        mask = np.array([[True, False, True], [False, True, False]], dtype=bool)
        result = project_mask_to_source(mask, {"matrix": np.eye(3).tolist()}, mask.shape)
        np.testing.assert_array_equal(result, mask)
        self.assertEqual(result.dtype, np.bool_)

    def test_crop_translation_does_not_extrapolate_outside_observed_grid(self):
        mask = np.ones((2, 4), dtype=bool)
        transform = {"matrix": [[1, 0, -1], [0, 1, -1], [0, 0, 1]]}
        result = project_mask_to_source(mask, transform, (4, 6))
        expected = np.zeros((4, 6), dtype=bool)
        expected[1:3, 1:5] = True
        np.testing.assert_array_equal(result, expected)

    def test_saved_pixel_center_resize_mapping_is_preserved(self):
        mask = np.array([[False, True, False], [True, False, True]], dtype=bool)
        transform = {"matrix": [[.5, 0, -.25], [0, .5, -.25], [0, 0, 1]]}
        result = project_mask_to_source(mask, transform, (4, 6))
        expected = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
        np.testing.assert_array_equal(result, expected)

    def test_padding_is_removed_by_saved_source_translation(self):
        mask = np.zeros((4, 6), dtype=bool)
        mask[1:3, 1:5] = [[True, False, False, True], [False, True, True, False]]
        transform = {"matrix": [[1, 0, 1], [0, 1, 1], [0, 0, 1]]}
        result = project_mask_to_source(mask, transform, (2, 4))
        np.testing.assert_array_equal(result, mask[1:3, 1:5])

    def test_exif_like_clockwise_rotation_uses_original_source_axes(self):
        mask = np.zeros((4, 3), dtype=bool)
        mask[0, 0] = mask[3, 2] = True
        transform = {"matrix": [[0, -1, 2], [1, 0, 0], [0, 0, 1]]}
        result = project_mask_to_source(mask, transform, (3, 4))
        expected = np.zeros((3, 4), dtype=bool)
        expected[2, 0] = expected[0, 3] = True
        np.testing.assert_array_equal(result, expected)

    def test_affine_shear_uses_saved_integer_pixel_centers(self):
        mask = np.zeros((4, 8), dtype=bool)
        mask[2, 4] = True
        transform = {"matrix": [[1, 1, 0], [0, 1, 0], [0, 0, 1]]}
        result = project_mask_to_source(mask, transform, (4, 4))
        expected = np.zeros((4, 4), dtype=bool)
        expected[2, 2] = True
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
