"""Small CPU fixtures, excluded from real experiment coverage."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from traversability_data.annotations import save_annotation
from traversability_data.masks import freeze_selection, import_sam_masks
from traversability_data.storage import atomic_json, atomic_jsonl, data_path, read_json, read_jsonl


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root, self.run_dir = self.root / "data", self.root / "runs" / "fixture"
        self.data_root.mkdir()
        self.frame_id = "fixture:dev:00000"
        self.frame = self.make_frame(self.frame_id, "fixture_rgb.png")
        atomic_jsonl(self.run_dir / "frames.jsonl", [self.frame])
        atomic_json(self.run_dir / "metadata" / "data.json", {
            "is_fixture": True, "robot_profile_id": "rellis_material_v1"
        })

    def make_frame(self, frame_id, image_path):
        Image.new("RGB", (20, 12), (40, 70, 100)).save(self.data_root / image_path)
        return {
            "frame_id": frame_id, "source": "fixture", "scene_id": "fixture-scene",
            "sequence_id": "fixture-dev", "timestamp_s": None,
            "image_path": image_path, "split": "development",
        }

    def record(self, region_id, area=3, *, frame_id=None, relevant=True, value=255):
        mask = np.zeros((12, 20), dtype=np.uint8)
        mask.flat[:area] = value
        source = self.root / (hashlib.sha256(region_id.encode()).hexdigest() + ".png")
        Image.fromarray(mask).save(source)
        return {
            "region_id": region_id, "frame_id": frame_id or self.frame_id,
            "mask_path": str(source), "planning_relevant": relevant,
        }

    def test_import_alignment_defaults_and_completed_annotations_survive_resume(self):
        record = self.record("fixture:dev:00000:r1", value=1)
        imported = import_sam_masks(self.run_dir, self.data_root, [record])
        self.assertFalse(imported[0]["selected_for_classification"])
        self.assertTrue(imported[0]["planning_relevant"])
        with Image.open(data_path(self.data_root, imported[0]["mask_path"])) as mask:
            self.assertEqual(mask.mode, "L")
            self.assertEqual(mask.size, (20, 12))
            self.assertEqual(set(np.unique(mask)), {0, 255})
        pending = read_jsonl(self.run_dir / "annotations.jsonl")[0]
        self.assertIsNone(pending["reference_label"])
        self.assertEqual(pending["annotation_status"], "pending")
        save_annotation(self.run_dir, imported[0]["region_id"], "non_traversable", "valid")
        completed_bytes = (self.run_dir / "annotations.jsonl").read_bytes()
        selection = freeze_selection(self.run_dir, self.data_root)
        repeated = import_sam_masks(self.run_dir, self.data_root, [record])
        self.assertTrue(repeated[0]["selected_for_classification"])
        self.assertEqual(selection, freeze_selection(self.run_dir, self.data_root))
        self.assertEqual(completed_bytes, (self.run_dir / "annotations.jsonl").read_bytes())
        provenance = read_json(self.run_dir / "metadata" / "masks.json")
        self.assertEqual(provenance[record["region_id"]]["origin"], "external_sam")

    def test_duplicate_ids_bad_dimensions_channels_semantic_values_and_origins_fail(self):
        record = self.record("fixture:dev:00000:r1")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            import_sam_masks(self.run_dir, self.data_root, [record, record])
        for image in (Image.new("L", (5, 5), 255), Image.new("RGB", (20, 12), "white")):
            image.save(record["mask_path"])
            with self.assertRaises(ValueError):
                import_sam_masks(self.run_dir, self.data_root, [record])
        values = np.zeros((12, 20), dtype=np.uint8)
        values.flat[0], values.flat[1] = 1, 2
        Image.fromarray(values).save(record["mask_path"])
        with self.assertRaisesRegex(ValueError, "foreground"):
            import_sam_masks(self.run_dir, self.data_root, [record])
        for origin in ("semantic", "dataset_semantic", "ground_truth", "reference"):
            with self.assertRaisesRegex(ValueError, "semantic/reference"):
                import_sam_masks(self.run_dir, self.data_root, [record], mask_origin=origin)
        self.assertFalse((self.run_dir / "regions.jsonl").exists())

    def test_synthetic_origin_requires_fixture_marker(self):
        record = self.record("fixture:dev:00000:r1")
        atomic_json(self.run_dir / "metadata" / "data.json", {"is_fixture": False})
        with self.assertRaisesRegex(ValueError, "fixture"):
            import_sam_masks(self.run_dir, self.data_root, [record], mask_origin="synthetic_fixture")
        atomic_json(self.run_dir / "metadata" / "data.json", {"is_fixture": True})
        import_sam_masks(self.run_dir, self.data_root, [record], mask_origin="synthetic_fixture")

    def test_known_semantic_aid_is_rejected_even_if_it_looks_binary(self):
        semantic_path = self.data_root / "semantic_ids.png"
        mask = np.zeros((12, 20), dtype=np.uint8)
        mask.flat[:3] = 1
        Image.fromarray(mask).save(semantic_path)
        atomic_json(self.run_dir / "metadata" / "data.json", {
            "is_fixture": True, "semantic_aids": {self.frame_id: "semantic_ids.png"}
        })
        record = {
            "region_id": "fixture:dev:00000:r1", "frame_id": self.frame_id,
            "mask_path": str(semantic_path),
        }
        with self.assertRaisesRegex(ValueError, "semantic aid"):
            import_sam_masks(self.run_dir, self.data_root, [record])
        self.assertFalse((self.run_dir / "regions.jsonl").exists())

    def test_reimport_conflicts_and_producer_rgb_hash_rejected(self):
        record = self.record("fixture:dev:00000:r1")
        with self.assertRaisesRegex(ValueError, "RGB hash"):
            import_sam_masks(self.run_dir, self.data_root, [{**record, "image_sha256": "bad"}])
        import_sam_masks(self.run_dir, self.data_root, [record])
        changed = self.record(record["region_id"], area=4)
        with self.assertRaisesRegex(ValueError, "Existing mask differs"):
            import_sam_masks(self.run_dir, self.data_root, [changed])

    def test_selection_deduplicates_preserves_all_and_uses_area_thirds_hash_ranks(self):
        records = [self.record(f"r{index:02d}", area=index) for index in range(1, 10)]
        records += [self.record("zzduplicate", area=3), self.record("ignored", area=20, relevant=False)]
        import_sam_masks(self.run_dir, self.data_root, list(reversed(records)))
        freeze = freeze_selection(self.run_dir, self.data_root, seed=4)
        details = freeze["frames"][self.frame_id]
        self.assertEqual(details["duplicate_of"], {"zzduplicate": "r03"})
        self.assertEqual(len(details["candidates"]), 11)
        self.assertEqual(len(read_jsonl(self.run_dir / "regions.jsonl")), 11)
        def rank(region_id):
            return hashlib.sha256(f"4\0{self.frame_id}\0{region_id}".encode()).hexdigest()
        expected = []
        for ids, count in ((["r01", "r02", "r03"], 2), (["r04", "r05", "r06"], 1), (["r07", "r08", "r09"], 2)):
            expected.extend(sorted(ids, key=lambda key: (rank(key), key))[:count])
        self.assertEqual(details["selected_region_ids"], sorted(expected))
        selected = [row for row in read_jsonl(self.run_dir / "regions.jsonl") if row["selected_for_classification"]]
        self.assertEqual(len(selected), 5)
        self.assertEqual(freeze, freeze_selection(self.run_dir, self.data_root, seed=4))
        with self.assertRaisesRegex(ValueError, "rule/seed"):
            freeze_selection(self.run_dir, self.data_root, seed=5)

    def test_all_five_or_fewer_unique_masks_selected_and_canonical_lowest_id(self):
        import_sam_masks(self.run_dir, self.data_root, [
            self.record("z", area=2), self.record("a", area=2), self.record("b", area=4)
        ])
        details = freeze_selection(self.run_dir, self.data_root)["frames"][self.frame_id]
        self.assertEqual(details["selected_region_ids"], ["a", "b"])
        self.assertEqual(details["duplicate_of"], {"z": "a"})

    def test_frozen_mask_rgb_relevance_and_flags_changes_rejected(self):
        import_sam_masks(self.run_dir, self.data_root, [self.record("a", area=3)])
        freeze_selection(self.run_dir, self.data_root)
        regions_path = self.run_dir / "regions.jsonl"
        originals = read_jsonl(regions_path)
        atomic_jsonl(regions_path, [{**originals[0], "selected_for_classification": False}])
        with self.assertRaisesRegex(ValueError, "flags changed"):
            freeze_selection(self.run_dir, self.data_root)
        atomic_jsonl(regions_path, originals)
        Image.new("RGB", (20, 12), "red").save(self.data_root / self.frame["image_path"])
        with self.assertRaisesRegex(ValueError, "Frozen frame/masks changed"):
            freeze_selection(self.run_dir, self.data_root)
        Image.new("RGB", (20, 12), (40, 70, 100)).save(self.data_root / self.frame["image_path"])
        mask_path = data_path(self.data_root, originals[0]["mask_path"])
        mask = np.zeros((12, 20), dtype=np.uint8)
        mask.flat[10:13] = 255
        Image.fromarray(mask).save(mask_path)
        with self.assertRaisesRegex(ValueError, "Frozen frame/masks changed"):
            freeze_selection(self.run_dir, self.data_root)

    def test_new_frame_masks_append_but_frozen_frame_candidates_cannot_change(self):
        second_id = "fixture:dev:00001"
        second = self.make_frame(second_id, "second.png")
        atomic_jsonl(self.run_dir / "frames.jsonl", [self.frame, second])
        import_sam_masks(self.run_dir, self.data_root, [self.record("a")])
        initial = freeze_selection(self.run_dir, self.data_root)
        self.assertEqual(initial["pending_frame_ids"], [second_id])
        with self.assertRaisesRegex(ValueError, "frozen frame"):
            import_sam_masks(self.run_dir, self.data_root, [self.record("new", area=4)])
        import_sam_masks(self.run_dir, self.data_root, [self.record("b", frame_id=second_id)])
        extended = freeze_selection(self.run_dir, self.data_root)
        self.assertEqual(initial["frames"][self.frame_id], extended["frames"][self.frame_id])
        self.assertEqual(extended["pending_frame_ids"], [])

    def test_freeze_includes_frame_ids_splits_timestamps_and_location(self):
        import_sam_masks(self.run_dir, self.data_root, [self.record("a")])
        frozen = freeze_selection(self.run_dir, self.data_root)
        self.assertEqual(frozen["frames"][self.frame_id]["frame_record"], self.frame)
        for field, changed_value in (
            ("timestamp_s", 0.25), ("scene_id", "different-scene"),
            ("sequence_id", "different-sequence"), ("split", "test"),
        ):
            with self.subTest(field=field):
                atomic_jsonl(self.run_dir / "frames.jsonl", [{**self.frame, field: changed_value}])
                with self.assertRaisesRegex(ValueError, "Frozen frame/masks changed"):
                    freeze_selection(self.run_dir, self.data_root)
                atomic_jsonl(self.run_dir / "frames.jsonl", [self.frame])
        self.assertEqual(frozen, freeze_selection(self.run_dir, self.data_root))


if __name__ == "__main__":
    unittest.main()
