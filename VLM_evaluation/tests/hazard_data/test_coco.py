"""Official CPU polygon/RLE semantics and prediction-independent selection."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from pycocotools import mask as mask_api

from common import COMPONENT, read_json, write_json
from traversability_hazard_data.coco import CocoDataset


CONCEPTS = ["person", "cup", "bottle", "chair", "dog"]
CATEGORY_IDS = {concept: 100 + index * 7 for index, concept in enumerate(CONCEPTS)}


def encoded(mask):
    rle = mask_api.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": rle["size"], "counts": rle["counts"].decode("ascii")}


def uncompressed(mask):
    flattened = mask.ravel(order="F")
    counts, value, length = [], 0, 0
    for pixel in flattened:
        if pixel == value:
            length += 1
        else:
            counts.append(length)
            value = int(pixel)
            length = 1
    counts.append(length)
    return {"size": list(mask.shape), "counts": counts}


def document(images=1, shape=(50, 60)):
    height, width = shape
    return {
        "images": [{"id": index, "file_name": f"{index:012d}.jpg",
                    "width": width, "height": height,
                    "coco_url": f"http://images.cocodataset.org/val2017/{index:012d}.jpg"}
                   for index in range(images)],
        "annotations": [],
        "categories": [{"id": CATEGORY_IDS[concept], "name": concept} for concept in CONCEPTS],
    }


class CocoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = read_json(COMPONENT / "configs/hazards/policy.json")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "instances_val2017.json"

    def tearDown(self):
        self.temporary.cleanup()

    def load(self, value):
        write_json(self.path, value)
        return CocoDataset(self.path, self.policy)

    def test_polygon_compressed_uncompressed_and_crowd_union(self):
        data = document()
        compressed = np.zeros((50, 60), dtype=np.uint8)
        compressed[2:6, 2:7] = 1
        crowd = np.zeros_like(compressed)
        crowd[4:8, 5:9] = 1
        one_pixel = np.zeros_like(compressed)
        one_pixel[45, 55] = 1
        polygon = [[1, 1, 5, 1, 5, 5, 1, 5]]
        annotations = [
            ("cup", polygon, 0), ("cup", encoded(compressed), 0),
            ("cup", uncompressed(crowd), 1), ("dog", encoded(one_pixel), 0),
            ("bottle", encoded(np.zeros_like(compressed)), 0),
        ]
        for index, (concept, segmentation, iscrowd) in enumerate(annotations):
            data["annotations"].append({"id": index, "image_id": 0,
                                        "category_id": CATEGORY_IDS[concept],
                                        "segmentation": segmentation, "iscrowd": iscrowd,
                                        "area": 999999, "bbox": [0, 0, 60, 50]})
        dataset = self.load(data)
        masks, valid, details = dataset.reference(0)
        expected_polygon = mask_api.decode(mask_api.merge(mask_api.frPyObjects(polygon, 50, 60)))
        np.testing.assert_array_equal(masks["cup"], (expected_polygon | compressed | crowd).astype(bool))
        self.assertEqual(masks["dog"].sum(), 1)
        self.assertIn("dog", details["tiny_concepts"])
        self.assertNotIn("bottle", masks)
        self.assertTrue(valid.all())
        self.assertEqual(details["ignored_pixel_count"], 0)
        self.assertEqual(dataset.category_ids, CATEGORY_IDS)
        selected = dataset.select(fixture=True)["selected"][0]
        self.assertEqual(selected["present_concepts"], ["cup", "dog"])
        self.assertEqual(selected["small_concepts"], ["cup", "dog"])

    def test_real_quota_selection_is_reproducible_balanced_and_disjoint(self):
        data = document(images=88)
        little = np.zeros((50, 60), dtype=np.uint8)
        little[1:3, 1:3] = 1
        for image_id in range(80):
            concept = CONCEPTS[image_id % len(CONCEPTS)]
            data["annotations"].append({"id": image_id, "image_id": image_id,
                                        "category_id": CATEGORY_IDS[concept],
                                        "segmentation": encoded(little), "iscrowd": 0})
        dataset = self.load(data)
        with mock.patch("pycocotools.mask.decode", side_effect=AssertionError("selection decodes rasters")):
            selection = dataset.select()
        reversed_data = copy.deepcopy(data)
        for key in ("images", "annotations", "categories"):
            reversed_data[key].reverse()
        self.assertEqual(self.load(reversed_data).select(), selection)
        self.assertEqual(dataset.select(), selection)
        ids = [row["image_id"] for row in selection["selected"]]
        self.assertEqual(len(ids), 40)
        self.assertEqual(len(set(ids)), 40)
        self.assertEqual(selection["missing_strata"], [])
        for split, positives, negatives in (("development", 6, 2), ("test", 26, 6)):
            rows = [row for row in selection["selected"] if row["split"] == split]
            self.assertEqual(sum(row["positive"] for row in rows), positives)
            self.assertEqual(sum(not row["positive"] for row in rows), negatives)
            self.assertEqual({concept for row in rows for concept in row["present_concepts"]}, set(CONCEPTS))
            self.assertEqual({concept for row in rows for concept in row["small_concepts"]}, set(CONCEPTS))

    def test_missing_strata_and_zero_masks_are_reported_without_inventing_ids(self):
        data = document(images=3)
        empty = np.zeros((50, 60), dtype=np.uint8)
        data["annotations"].append({"id": 1, "image_id": 1, "category_id": CATEGORY_IDS["cup"],
                                    "segmentation": encoded(empty), "iscrowd": 1})
        selected = self.load(data).select()
        self.assertEqual(len(selected["selected"]), 3)
        self.assertTrue(all(not row["positive"] for row in selected["selected"]))
        self.assertTrue(any(row["stratum"] == "positive_images" for row in selected["missing_strata"]))
        self.assertTrue(any(row.get("concept") == "cup" for row in selected["missing_strata"]))

    def test_category_resolution_duplicate_ids_bad_joins_and_escaped_names(self):
        cases = []
        missing = document()
        missing["categories"].pop()
        cases.append(missing)
        duplicate = document()
        duplicate["images"].append(copy.deepcopy(duplicate["images"][0]))
        cases.append(duplicate)
        bad_join = document()
        bad_join["annotations"].append({"id": 1, "image_id": 100,
                                        "category_id": CATEGORY_IDS["person"], "segmentation": []})
        cases.append(bad_join)
        for filename in ("../x.jpg", "C:/x.jpg", "a\\x.jpg", "/x.jpg", "a/../x.jpg",
                         "CON.jpg", "a?.jpg", "folder/trailing."):
            escaped = document()
            escaped["images"][0]["file_name"] = filename
            cases.append(escaped)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.load(data)

    def test_malformed_rle_polygon_and_dimensions_are_rejected(self):
        for segmentation in ({"size": [50, 60], "counts": [3001]},
                             {"size": [50, 60], "counts": [3000, -1, 1]},
                             {"size": [60, 50], "counts": [3000]},
                             {"size": [50, 60], "counts": "invalid"},
                             [[0, 0, 1, 1]], None):
            with self.subTest(segmentation=segmentation):
                data = document()
                data["annotations"].append({"id": 1, "image_id": 0,
                                            "category_id": CATEGORY_IDS["person"],
                                            "segmentation": segmentation})
                with self.assertRaises(ValueError):
                    self.load(data).select()


if __name__ == "__main__":
    unittest.main()
