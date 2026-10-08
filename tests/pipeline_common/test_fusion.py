"""CPU conformance checks for surface and independent semantic evidence."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline_common.contracts import FramePacket, GeometryFrame, InstanceRecord, QueryRecord, SemanticFrame
from pipeline_common.fusion import VoxelFuser


def frame(frame_id="frame0", timestamp=1000, *, points=None, confidence=None):
    points = np.asarray(points if points is not None else [[0.01, 0.01, 0.01],
        [0.02, 0.01, 0.01], [0.31, 0.01, 0.01]], dtype=np.float64).reshape(1, -1, 3)
    shape = points.shape[:2]
    rgb = np.arange(np.prod(points.shape), dtype=np.uint8).reshape(points.shape)
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [1, -2, 0.5]
    geometry = GeometryFrame(points=points, depth=np.ones(shape), validity=np.ones(shape, dtype=bool),
        confidence=np.asarray(confidence if confidence is not None else [2, 4, 3], dtype=float).reshape(shape),
        intrinsics=np.array([[40., 0, 1], [0, 30, 0], [0, 0, 1]]), world_to_camera=pose,
        processed_grid_id="grid1", geometry_fingerprint="verified-cache1:" + frame_id,
        map_frame="fixture_world", units="metres", up=(0., 0., 1.),
        scale={"verified": True, "factor": 2.0, "source": "independent synthetic calibration"},
        pose_revision="revision0")
    return FramePacket(sequence_id="synthetic", frame_id=frame_id, timestamp_ns=timestamp,
        timestamp_provenance={"clock": "fixture_capture", "synthetic": True}, rgb=rgb,
        image_path=Path(frame_id + ".png"), encoded_file_sha256="encoded-file-identity:" + frame_id,
        decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), processed_grid_id="grid1",
        source_rgb_identity={"source_grid_id": "source", "source_dimensions": [7, 10]},
        source_to_processed={"crop": [2, 3, 3, 1], "resize": [3, 1]}, geometry=geometry)


def query(concept="hazard.a", *, query_id=None, phrase=None, mask=None, score=0.8,
        status="ok", role="hazard", grid="grid1", mapping_version="v1"):
    identifier = query_id or concept + ":query"
    instances = [] if mask is None else [InstanceRecord(identifier + ":observation",
        np.asarray(mask, dtype=bool).reshape(1, -1), score, "SAM raw evidence", grid)]
    return QueryRecord(identifier, phrase or concept, concept, role, status,
        None if status == "ok" else {"message": "synthetic text failure"}, instances,
        mapping_version=mapping_version)


def semantics(packet, queries=(), *, status="ok"):
    return SemanticFrame(sequence_id=packet.sequence_id, frame_id=packet.frame_id,
        processed_grid_id=packet.processed_grid_id, decoded_rgb_sha256=packet.decoded_rgb_sha256,
        geometry_fingerprint=packet.geometry.geometry_fingerprint if packet.geometry else None,
        adapter_provenance={"fixture": True, "adapter": "unit-fixture"}, status=status,
        queries=list(queries), model_calls={} if status == "not_applicable" else {"sam3": 1},
        query_count=len(queries), timestamp_ns=packet.timestamp_ns)


class FusionTests(unittest.TestCase):
    def test_geometry_only_numeric_schemas_and_surface_states(self):
        fuser = VoxelFuser({"voxel_size": .25})
        packet = frame()
        fuser.add_geometry(packet)
        fuser.add_semantics(semantics(packet, status="not_applicable"), packet)
        voxels, evidence, concepts, journal, metadata = fuser.export()
        np.testing.assert_array_equal(voxels["voxel_indices"], [[0, 0, 0], [1, 0, 0]])
        np.testing.assert_allclose(voxels["centers"], [[.125, .125, .125], [.375, .125, .125]])
        np.testing.assert_array_equal(voxels["geometry_weight"], [4, 3])
        np.testing.assert_array_equal(voxels["frame_support"], [1, 1])
        np.testing.assert_array_equal(voxels["observation_state"], [2, 2])
        self.assertEqual(concepts, [])
        self.assertEqual(len(journal), 1)
        self.assertFalse(metadata["free_space_capability"])
        self.assertEqual(metadata["up"], [0, 0, 1])
        self.assertEqual(metadata["units"], "metres")
        for name, array in evidence.items():
            self.assertEqual(array.shape, (0,), name)
            self.assertNotEqual(array.dtype.kind, "O", name)
        for name in ("voxel_row", "concept_row", "frame_support", "last_seen_timestamp_ns"):
            self.assertEqual(evidence[name].dtype, np.int64)
        for name in ("positive_weight", "observed_weight", "evidence_score"):
            self.assertEqual(evidence[name].dtype, np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry_only_evidence.npz"
            np.savez_compressed(path, **evidence)
            with np.load(path, allow_pickle=False) as restored:
                self.assertEqual(set(restored.files), set(evidence))
        json.dumps(journal, allow_nan=False)

    def test_correlated_pixels_and_aliases_capped_overlap_independent(self):
        packet = frame()
        fuser = VoxelFuser()
        alias = query(query_id="alias", phrase="original Qwen phrase", mask=[True, False, False], score=.6)
        higher = query(mask=[True, False, False], score=.8)
        overlapping = query("hazard.b", mask=[True, True, False], score=.5)
        fuser.add_semantics(semantics(packet, [higher, alias, overlapping]), packet)
        voxels, evidence, concepts, journal, metadata = fuser.export()
        self.assertEqual([c["concept_id"] for c in concepts], ["hazard.a", "hazard.b"])
        self.assertEqual(concepts[0]["original_phrases"], ["hazard.a", "original Qwen phrase"])
        np.testing.assert_allclose(evidence["positive_weight"], [1.6, 2, 0, 0])
        np.testing.assert_allclose(evidence["observed_weight"], [4, 4, 3, 3])
        np.testing.assert_allclose(evidence["evidence_score"], [.4, .5, 0, 0])
        np.testing.assert_array_equal(evidence["frame_support"], [1, 1, 1, 1])
        accepted = [row for row in journal if row["kind"] == "semantic"]
        self.assertEqual(len(accepted), 2)
        self.assertEqual(accepted[0]["query_ids"], [higher.query_id, "alias"])
        self.assertIn("not probability", metadata["evidence_score_definition"])
        np.testing.assert_array_equal(voxels["observation_state"], [2, 2])

    def test_empty_query_negative_only_queried_concept_omission_no_negative(self):
        first, second = frame(), frame("frame1", 2000)
        fuser = VoxelFuser()
        fuser.add_semantics(semantics(first, [query(mask=[True, True, True])]), first)
        # Qwen discovered no phrases: the old concept is unqueried on frame1.
        fuser.add_semantics(semantics(second), second)
        _, before, _, _, _ = fuser.export()
        np.testing.assert_array_equal(before["frame_support"], [1, 1])
        third = frame("frame2", 3000)
        fuser.add_semantics(semantics(third, [query()]), third)
        voxels, after, _, _, _ = fuser.export()
        np.testing.assert_array_equal(after["frame_support"], [2, 2])
        np.testing.assert_allclose(after["evidence_score"], [.4, .4])
        np.testing.assert_array_equal(voxels["frame_support"], [3, 3])
        np.testing.assert_array_equal(voxels["observation_state"], [2, 2])

    def test_failed_partial_masks_are_diagnostic_only(self):
        packet = frame()
        fuser = VoxelFuser()
        failed = query("hazard.failed", mask=[True, True, True], status="error", score=1)
        fuser.add_semantics(semantics(packet, [query(mask=[True, False, False]), failed], status="partial"), packet)
        _, evidence, concepts, journal, _ = fuser.export()
        self.assertEqual([c["concept_id"] for c in concepts], ["hazard.a", "hazard.failed"])
        self.assertEqual(set(evidence["concept_row"]), {0})
        failure = [row for row in journal if row.get("query_id") == failed.query_id][0]
        self.assertFalse(failure["fused"])
        self.assertEqual(failure["status"], "error")
        self.assertEqual(failure["observation_ids"], [failed.instances[0].observation_id])
        self.assertIsInstance(failure["mask_sha256"][0], str)

    def test_negative_evidence_can_be_disabled_without_certifying_empty_voxels(self):
        packet = frame()
        fuser = VoxelFuser({"negative_evidence": False})
        fuser.add_semantics(semantics(packet, [query(mask=[True, False, False]), query("hazard.empty")]), packet)
        voxels, evidence, concepts, _, metadata = fuser.export()
        np.testing.assert_array_equal(evidence["voxel_row"], [0])
        np.testing.assert_allclose(evidence["evidence_score"], [.8])
        self.assertEqual(len(concepts), 2)
        self.assertFalse(metadata["negative_evidence"])
        self.assertEqual(len(voxels["centers"]), 2)

    def test_identical_replay_idempotent_and_raw_points_not_mutable(self):
        packet = frame()
        result = semantics(packet, [query(mask=[True, False, False])])
        fuser = VoxelFuser()
        fuser.add_geometry(packet)
        fuser.add_semantics(result, packet)
        before = fuser.export()
        for _ in range(3):
            fuser.add_semantics(result, packet)
            fuser.add_geometry(packet)
        after = fuser.export()
        for index in (0, 1):
            for key in before[index]:
                np.testing.assert_array_equal(before[index][key], after[index][key])
        self.assertEqual(before[2:], after[2:])
        copied = fuser.observed_points()
        copied[:] = 99
        np.testing.assert_array_equal(fuser.observed_points(), packet.geometry.points.reshape(-1, 3))
        self.assertEqual(len(fuser.points), 3)

    def test_changed_pose_cache_or_payload_replay_rejected(self):
        packet = frame()
        result = semantics(packet, [query(mask=[True, False, False])])
        fuser = VoxelFuser()
        fuser.add_semantics(result, packet)
        changed_pose = packet.geometry.world_to_camera.copy()
        changed_pose[0, 3] += .25
        variants = [replace(packet.geometry, pose_revision="revision1"),
            replace(packet.geometry, geometry_fingerprint="new-cache"),
            replace(packet.geometry, world_to_camera=changed_pose),
            replace(packet.geometry, points=packet.geometry.points + .01)]
        for geometry in variants:
            with self.subTest(geometry=geometry.geometry_fingerprint, pose=geometry.pose_revision):
                with self.assertRaises(ValueError):
                    fuser.add_geometry(replace(packet, geometry=geometry))
        changed_result = semantics(packet, [query(mask=[False, True, False])])
        with self.assertRaisesRegex(ValueError, "Replayed semantic"):
            fuser.add_semantics(changed_result, packet)
        self.assertEqual(len(fuser.observed_points()), 3)

    def test_units_map_up_scale_and_revision_cannot_mix(self):
        first, second = frame(), frame("frame1")
        for changes in ({"units": "reconstruction_units"}, {"map_frame": "different"},
                {"up": (0, 1, 0)}, {"scale": {"verified": False}}, {"pose_revision": "revision1"}):
            fuser = VoxelFuser()
            fuser.add_geometry(first)
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "map frame/units/up/scale/pose"):
                    fuser.add_geometry(replace(second, geometry=replace(second.geometry, **changes)))
            self.assertEqual(fuser.export()[4]["frame_count"], 1)

    def test_exact_join_rgb_and_mask_grids_validated(self):
        packet = frame()
        result = semantics(packet, [query(mask=[True, False, False])])
        for changes in ({"sequence_id": "wrong"}, {"frame_id": "wrong"},
                {"processed_grid_id": "source"}, {"decoded_rgb_sha256": "wrong"},
                {"geometry_fingerprint": "wrong"}, {"timestamp_ns": 123}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    VoxelFuser().add_semantics(replace(result, **changes), packet)
        bad_rgb = packet.rgb.copy()
        bad_rgb[0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "Decoded RGB"):
            VoxelFuser().add_geometry(replace(packet, rgb=bad_rgb))
        with self.assertRaisesRegex(ValueError, "Geometry grid"):
            VoxelFuser().add_geometry(replace(packet, geometry=replace(packet.geometry, processed_grid_id="source")))
        for instance in (replace(result.queries[0].instances[0], mask=np.ones((2, 3), dtype=bool)),
                replace(result.queries[0].instances[0], mask=np.ones((1, 3), dtype=np.uint8)),
                replace(result.queries[0].instances[0], processed_grid_id="source")):
            bad = replace(result, queries=[replace(result.queries[0], instances=[instance])])
            with self.assertRaises(ValueError):
                VoxelFuser().add_semantics(bad, packet)

    def test_invalid_depth_confidence_points_or_validity_do_not_observe_space(self):
        packet = frame(points=[[x, 0, 0] for x in range(7)], confidence=[2] * 7)
        geometry = packet.geometry
        geometry.depth[0, 0] = 0
        geometry.depth[0, 1] = np.nan
        geometry.confidence[0, 2] = -1
        geometry.confidence[0, 3] = np.inf
        geometry.points[0, 4] = [np.nan, 0, 0]
        geometry.validity[0, 5] = False
        fuser = VoxelFuser()
        fuser.add_semantics(semantics(packet, [query(mask=[True] * 7)]), packet)
        voxels, evidence, _, _, _ = fuser.export()
        np.testing.assert_array_equal(voxels["voxel_indices"], [[24, 0, 0]])
        self.assertEqual(len(evidence["voxel_row"]), 1)
        self.assertEqual(fuser.observed_points().tolist(), [[6, 0, 0]])
        self.assertEqual(voxels["observation_state"].tolist(), [2])

    def test_empty_or_missing_geometry_and_unknown_timestamps_numeric(self):
        packet = frame(timestamp=None)
        fuser = VoxelFuser()
        packet.geometry.depth[:] = 0
        fuser.add_semantics(semantics(packet, [query()]), packet)
        voxels, evidence, _, _, _ = fuser.export()
        self.assertEqual(voxels["voxel_indices"].shape, (0, 3))
        self.assertEqual(voxels["centers"].shape, (0, 3))
        self.assertEqual(len(evidence["voxel_row"]), 0)
        missing = replace(frame("missing", None), geometry=None)
        fuser.add_semantics(semantics(missing, [query()]), missing)
        self.assertEqual(len(fuser.observed_points()), 0)
        known_geometry = frame("unclocked", None)
        fuser.add_semantics(semantics(known_geometry, [query()]), known_geometry)
        voxels, evidence, _, _, _ = fuser.export()
        np.testing.assert_array_equal(voxels["last_observed_timestamp_ns"], [-1, -1])
        np.testing.assert_array_equal(evidence["last_seen_timestamp_ns"], [-1, -1])

    def test_expiry_filters_old_contributions_without_clearing_surfaces(self):
        first, second = frame(), frame("frame1", 2000)
        fuser = VoxelFuser({"semantic_expiry_ns": 500})
        fuser.add_semantics(semantics(first, [query(mask=[True, True, True])]), first)
        fuser.add_semantics(semantics(second, [query()]), second)
        voxels, evidence, _, journal, metadata = fuser.export(2100)
        np.testing.assert_array_equal(evidence["positive_weight"], [0, 0])
        np.testing.assert_array_equal(evidence["frame_support"], [1, 1])
        np.testing.assert_array_equal(voxels["frame_support"], [2, 2])
        self.assertEqual(metadata["stale_semantic_contributions"], 1)
        self.assertEqual({row["evidence_state"] for row in journal if row["kind"] == "semantic"}, {"active", "stale"})
        expired = fuser.export(3000)
        self.assertEqual(len(expired[1]["voxel_row"]), 0)
        self.assertEqual(len(expired[0]["voxel_indices"]), 2)
        self.assertEqual(expired[4]["stale_semantic_contributions"], 2)
        # Export filtering does not delete evidence or mutate the contribution log.
        self.assertEqual(len(fuser.export(2100)[1]["voxel_row"]), 2)
        with self.assertRaisesRegex(ValueError, "explicit capture-clock"):
            fuser.export()
        with self.assertRaisesRegex(ValueError, "precedes"):
            fuser.export(1500)

    def test_unknown_expiry_age_omits_semantics_and_preserves_unknown_provenance(self):
        packet = frame(timestamp=None)
        fuser = VoxelFuser({"semantic_expiry_ns": 500})
        fuser.add_semantics(semantics(packet, [query(mask=[True, True, True])]), packet)
        voxels, evidence, _, journal, metadata = fuser.export(1000)
        self.assertEqual(len(evidence["voxel_row"]), 0)
        self.assertEqual(len(voxels["voxel_indices"]), 2)
        self.assertEqual(metadata["unknown_time_semantic_contributions"], 1)
        self.assertEqual([row["evidence_state"] for row in journal if row["kind"] == "semantic"], ["unknown_capture_time"])

    def test_deterministic_registry_negative_indices_and_row_identity(self):
        packet = frame(points=[[-.001, 0, 0], [.251, 0, 0], [.001, 0, 0]])
        fuser = VoxelFuser()
        fuser.add_semantics(semantics(packet, [query("z", mask=[True, False, False]),
            query("a", mask=[False, True, False])]), packet)
        voxels, evidence, concepts, _, _ = fuser.export()
        self.assertEqual([entry["concept_id"] for entry in concepts], ["a", "z"])
        np.testing.assert_array_equal(voxels["voxel_indices"], [[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
        positive = evidence["positive_weight"] > 0
        self.assertEqual([(voxels["voxel_indices"][v].tolist(), concepts[c]["concept_id"])
            for v, c in zip(evidence["voxel_row"][positive], evidence["concept_row"][positive])],
            [([-1, 0, 0], "z"), ([1, 0, 0], "a")])

    def test_concept_role_or_mapping_conflicts_rejected_before_geometry_mutation(self):
        first, second = frame(), frame("frame1")
        fuser = VoxelFuser()
        fuser.add_semantics(semantics(first, [query()]), first)
        for changes in ({"role": "candidate_surface"}, {"mapping_version": "different"}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "Concept role/mapping"):
                    fuser.add_semantics(semantics(second, [query(**changes)]), second)
        self.assertEqual(fuser.export()[4]["frame_count"], 1)

    def test_configuration_pose_and_overflow_rejections(self):
        for config in ({"voxel_size": 0}, {"voxel_size": float("nan")}, {"voxel_size": True},
                {"origin": [0, 0]}, {"origin": [0, 0, float("inf")]},
                {"semantic_expiry_ns": -1}, {"negative_evidence": 1}, {"min_point_confidence": -1}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    VoxelFuser(config)
        packet = frame()
        invalid_pose = packet.geometry.world_to_camera.copy()
        invalid_pose[0, 0] = 4
        with self.assertRaisesRegex(ValueError, "rotation"):
            VoxelFuser().add_geometry(replace(packet, geometry=replace(packet.geometry, world_to_camera=invalid_pose)))
        huge = replace(packet, geometry=replace(packet.geometry, points=np.full((1, 3, 3), 1e100)))
        with self.assertRaisesRegex(ValueError, "int64"):
            VoxelFuser().add_geometry(huge)


if __name__ == "__main__":
    unittest.main()
