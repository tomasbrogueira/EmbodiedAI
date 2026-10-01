"""Public API COCO freezing/fill behavior on annotation-only synthetic inputs."""

from collections import Counter
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from common import config, file_snapshot, read_json, read_jsonl, write_json
from traversability_hazard_data import prepare_run, validate_run
from traversability_hazard_data.storage import fingerprint


def synthetic_coco_inputs(base):
    base = Path(base)
    names = ["person", "cup", "bottle", "chair", "dog", "car"]
    categories = [{"id": 10 + index, "name": name} for index, name in enumerate(names)]
    images = [{"id": number, "width": 16, "height": 12, "file_name": f"image_{number}.png",
               "coco_url": f"https://images.cocodataset.org/val2017/image_{number}.png"}
              for number in range(1, 49)]
    annotations = []
    for number in range(1, 49):
        # Last eight controls contain an unscored car, so absence is vocabulary-specific.
        category = 10 + ((number - 1) % 5) if number <= 40 else 15
        annotations.append({"id": number, "image_id": number, "category_id": category,
                            "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                            "area": 9, "bbox": [1, 1, 3, 3], "iscrowd": 0})
    annotation = base / "annotations.json"
    write_json(annotation, {"images": images, "categories": categories, "annotations": annotations})
    image_dir = base / "images"
    image_dir.mkdir()
    return annotation, image_dir, images


def supply_image(directory, image):
    values = np.zeros((image["height"], image["width"], 3), dtype=np.uint8)
    values[:, :, 0] = image["id"]
    values[:, :, 1] = np.arange(image["width"], dtype=np.uint8)
    values[:, :, 2] = np.arange(image["height"], dtype=np.uint8)[:, None]
    Image.fromarray(values).save(Path(directory) / image["file_name"])


class CocoPreparationTests(unittest.TestCase):
    def test_annotation_selected_ids_are_frozen_before_rgb_and_only_frozen_ids_fill(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            annotation, image_dir, images = synthetic_coco_inputs(base)
            cfg = config(base)
            cfg["coco"] = {"annotation_path": str(annotation), "image_dir": str(image_dir)}
            run = prepare_run(cfg)
            first = read_json(run / "metadata/dataset.json")
            selected = copy.deepcopy(first["selection"]["coco"]["selected"])
            self.assertEqual(len(selected), 40)
            self.assertEqual(read_jsonl(run / "frames.jsonl"), [])
            self.assertEqual(Counter(row["split"] for row in selected), {"development": 8, "test": 32})
            self.assertEqual(len({row["image_id"] for row in selected}), 40)
            # A pending run published before integration acquires its pin when
            # new inputs fill, without retroactively rewriting unchanged runs.
            del first["observed_input_signature"], first["selected_frame_ids"]
            first["dataset_fingerprint"] = fingerprint(first, [], [])
            write_json(run / "metadata/dataset.json", first)
            for image in images:
                supply_image(image_dir, image)
            prepare_run(cfg)
            final = read_json(run / "metadata/dataset.json")
            self.assertEqual(final["selection"]["coco"]["selected"], selected)
            frames = read_jsonl(run / "frames.jsonl")
            self.assertEqual(len(frames), 40)
            self.assertEqual(final["selected_frame_ids"], [row["frame_id"] for row in frames])
            self.assertEqual(len(final["observed_input_signature"]), 64)
            self.assertEqual({int(row["frame_id"].split(":")[-1]) for row in frames},
                             {row["image_id"] for row in selected})
            references = read_jsonl(run / "references.jsonl")
            by_frame = {row["frame_id"]: row for row in frames}
            counts = Counter((by_frame[row["frame_id"]]["split"], bool(row["present_concepts"]))
                             for row in references)
            self.assertEqual(counts, {("development", True): 6, ("development", False): 2,
                                      ("test", True): 26, ("test", False): 6})
            for row in references:
                self.assertTrue(row["absence_scoring_eligible"])
                self.assertNotIn("car", row["present_concepts"])
            self.assertTrue(validate_run(run, cfg["data_root"])["valid"])
            before = file_snapshot(run)
            prepare_run(cfg)
            self.assertEqual(file_snapshot(run), before)

    def test_changed_annotation_bytes_are_refused_before_missing_rgb_can_fill(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            annotation, image_dir, _ = synthetic_coco_inputs(base)
            cfg = config(base)
            cfg["coco"] = {"annotation_path": str(annotation), "image_dir": str(image_dir)}
            run = prepare_run(cfg)
            before = file_snapshot(run)
            annotation.write_bytes(annotation.read_bytes() + b"\n")
            with self.assertRaises((ValueError, OSError)):
                prepare_run(cfg)
            self.assertEqual(file_snapshot(run), before)

    def test_downstream_prediction_blocks_missing_selected_rgb_addition(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            annotation, image_dir, images = synthetic_coco_inputs(base)
            cfg = config(base)
            cfg["coco"] = {"annotation_path": str(annotation), "image_dir": str(image_dir)}
            run = prepare_run(cfg)
            selected = read_json(run / "metadata/dataset.json")["selection"]["coco"]["selected"]
            first_id = selected[0]["image_id"]
            supply_image(image_dir, next(row for row in images if row["id"] == first_id))
            (run / "predictions").mkdir()
            (run / "predictions/model.jsonl").write_text("{}\n", encoding="utf-8")
            before = file_snapshot(run)
            with self.assertRaises((ValueError, OSError)):
                prepare_run(cfg)
            self.assertEqual(file_snapshot(run), before)
            self.assertEqual(read_jsonl(run / "frames.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
