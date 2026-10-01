"""Annotation-only, reproducible COCO image selection and concept unions.

The official COCO mask API encodes polygons/RLE, computes areas, and decodes
selected masks. Bounding boxes and advertised annotation areas are never used.
"""

from collections import defaultdict
from fractions import Fraction
import hashlib
import math
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from .storage import hash_file, read_json, safe_name


def safe_image_name(value):
    """Require a portable, root-relative image filename, before any I/O."""
    if (not isinstance(value, str) or not value or "\\" in value
            or "\0" in value or PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).drive
            or any(part in ("", ".", "..") or ":" in part
                   for part in value.split("/"))):
        raise ValueError(f"Unsafe COCO file_name: {value!r}")
    for part in value.split("/"):
        safe_name(part)
    return value


def _integer(value, description, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{description} must be an integer >= {minimum}")
    return value


def _mask_api():
    try:
        from pycocotools import mask
    except ImportError as error:
        raise ImportError(
            "COCO references require the official CPU pycocotools mask API; "
            "install requirements/hazard_data.txt."
        ) from error
    return mask


def _compressed_counts(value):
    """Validate compressed runs before passing them to the native mask API.

    This only validates COCO's encoded run lengths; all mask operations still
    use pycocotools. It prevents malformed runs reaching a native decoder.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("Compressed COCO RLE counts must be a nonempty string")
    runs, index = [], 0
    while index < len(value):
        number, shift = 0, 0
        while True:
            if index >= len(value):
                raise ValueError("Truncated compressed COCO RLE counts")
            char = ord(value[index]) - 48
            index += 1
            if char < 0 or char > 63 or shift > 60:
                raise ValueError("Invalid compressed COCO RLE counts")
            number |= (char & 0x1F) << shift
            shift += 5
            if not char & 0x20:
                if char & 0x10:
                    number |= -1 << shift
                break
        if len(runs) > 2:
            number += runs[-2]
        if number < 0:
            raise ValueError("Negative compressed COCO RLE run length")
        runs.append(number)
    return runs


class CocoDataset:
    """Strict COCO instances schema with lazily cached encoded target masks."""

    def __init__(self, annotation_path, policy):
        self.annotation_path = Path(annotation_path).resolve()
        self.annotation_sha256 = hash_file(self.annotation_path)
        self.policy = policy
        self.concepts = list(policy["datasets"]["coco"]["scored_category_names"])
        if (len(self.concepts) != len(set(self.concepts))
                or set(self.concepts) != {"person", "cup", "bottle", "chair", "dog"}):
            raise ValueError("COCO policy must declare the five frozen category names")
        document = read_json(self.annotation_path)
        if not isinstance(document, dict):
            raise ValueError("COCO instances annotations must be a JSON object")
        for name in ("images", "annotations", "categories"):
            if not isinstance(document.get(name), list):
                raise ValueError(f"COCO {name} must be a list")

        category_names, category_ids = {}, set()
        for category in document["categories"]:
            if not isinstance(category, dict):
                raise ValueError("COCO category must be an object")
            identifier = _integer(category.get("id"), "COCO category ID")
            name = category.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("COCO category name must be a nonempty string")
            if identifier in category_ids or name in category_names:
                raise ValueError(f"Duplicate COCO category ID/name: {identifier}/{name}")
            category_ids.add(identifier)
            category_names[name] = identifier
        missing = [name for name in self.concepts if name not in category_names]
        if missing:
            raise ValueError(f"COCO category table is missing scored names: {missing}")
        self.category_ids = {name: category_names[name] for name in self.concepts}
        self._concept_by_id = {identifier: name
                               for name, identifier in self.category_ids.items()}

        self.images = {}
        image_names = set()
        for image in document["images"]:
            if not isinstance(image, dict):
                raise ValueError("COCO image must be an object")
            identifier = _integer(image.get("id"), "COCO image ID")
            _integer(image.get("width"), "COCO image width", minimum=1)
            _integer(image.get("height"), "COCO image height", minimum=1)
            filename = safe_image_name(image.get("file_name"))
            if identifier in self.images or filename in image_names:
                raise ValueError(f"Duplicate COCO image ID/file_name: {identifier}/{filename}")
            self.images[identifier] = dict(image)
            image_names.add(filename)

        self._annotations = defaultdict(list)
        annotation_ids = set()
        for annotation in document["annotations"]:
            if not isinstance(annotation, dict):
                raise ValueError("COCO annotation must be an object")
            identifier = _integer(annotation.get("id"), "COCO annotation ID")
            image_id = _integer(annotation.get("image_id"), "COCO annotation image ID")
            category_id = _integer(annotation.get("category_id"), "COCO annotation category ID")
            if identifier in annotation_ids:
                raise ValueError(f"Duplicate COCO annotation ID: {identifier}")
            annotation_ids.add(identifier)
            if image_id not in self.images or category_id not in category_ids:
                raise ValueError(f"COCO annotation {identifier} has an invalid image/category join")
            if (type(annotation.get("iscrowd", 0)) is not int
                    or annotation.get("iscrowd", 0) not in (0, 1)):
                raise ValueError(f"COCO annotation {identifier} has invalid iscrowd")
            if category_id in self._concept_by_id:
                self._annotations[image_id].append(dict(annotation))
        for rows in self._annotations.values():
            rows.sort(key=lambda annotation: annotation["id"])
        self._encoded = {}

    def _rle(self, annotation, image):
        api = _mask_api()
        height, width = image["height"], image["width"]
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list):
            if not segmentation:
                return None
            for polygon in segmentation:
                if (not isinstance(polygon, list) or len(polygon) < 6
                        or len(polygon) % 2
                        or any(type(value) not in (int, float) or not math.isfinite(value)
                               for value in polygon)):
                    raise ValueError(f"Invalid COCO polygon on annotation {annotation['id']}")
            return api.merge(api.frPyObjects(segmentation, height, width))
        if isinstance(segmentation, dict):
            if (segmentation.get("size") != [height, width]
                    or any(type(value) is not int for value in segmentation.get("size", []))):
                raise ValueError(f"COCO RLE dimensions differ on annotation {annotation['id']}")
            counts = segmentation.get("counts")
            if isinstance(counts, list):
                if not counts or any(type(value) is not int or value < 0 for value in counts):
                    raise ValueError(f"Invalid COCO RLE counts on annotation {annotation['id']}")
                runs = counts
            else:
                runs = _compressed_counts(counts)
            if sum(runs) != height * width:
                raise ValueError(f"COCO RLE runs do not cover image on annotation {annotation['id']}")
            rle = (api.frPyObjects(segmentation, height, width) if isinstance(counts, list)
                   else {"size": [height, width], "counts": counts.encode("ascii")})
            return rle
        raise ValueError(f"Missing/invalid COCO segmentation on annotation {annotation['id']}")

    def _image_masks(self, image_id):
        if image_id not in self.images:
            raise ValueError(f"Unknown COCO image ID: {image_id}")
        if image_id in self._encoded:
            return self._encoded[image_id]
        api = _mask_api()
        image = self.images[image_id]
        by_concept, small_concepts = defaultdict(list), set()
        for annotation in self._annotations[image_id]:
            concept = self._concept_by_id[annotation["category_id"]]
            rle = self._rle(annotation, image)
            if rle is None:
                continue
            area = int(api.area(rle))
            if area < 0 or area > image["width"] * image["height"]:
                raise ValueError(f"Invalid COCO mask area on annotation {annotation['id']}")
            if area > 0:
                by_concept[concept].append(rle)
                if area < 32 * 32:
                    small_concepts.add(concept)
        masks = {concept: api.merge(by_concept[concept])
                 for concept in self.concepts if by_concept[concept]}
        result = (masks, small_concepts)
        self._encoded[image_id] = result
        return result

    def reference(self, image_id):
        """Return original-size positive unions, all-valid mask, and diagnostics."""
        encoded, _ = self._image_masks(image_id)
        api = _mask_api()
        image = self.images[image_id]
        masks = {concept: np.asarray(api.decode(rle), dtype=bool)
                 for concept, rle in encoded.items()}
        shape = (image["height"], image["width"])
        if any(mask.shape != shape for mask in masks.values()):
            raise ValueError(f"COCO mask dimensions differ from image {image_id}")
        pixels = image["height"] * image["width"]
        fractions = {concept: int(mask.sum()) / pixels for concept, mask in masks.items()}
        tiny_limit = self.policy["tiny_concept_area_fraction"]
        details = {
            "concept_area_fractions": fractions,
            "tiny_concepts": [concept for concept in self.concepts
                              if 0 < fractions.get(concept, 0) < tiny_limit],
            "ignored_pixel_count": 0,
            "ignored_fraction": 0.0,
        }
        return masks, np.ones(shape, dtype=bool), details

    def select(self, *, fixture=False):
        """Freeze annotation-derived IDs; selection never inspects available RGB."""
        budgets = ({"development": {"positive": 2, "negative": 1},
                    "test": {"positive": 2, "negative": 1}} if fixture else
                   {"development": {"positive": 6, "negative": 2},
                    "test": {"positive": 26, "negative": 6}})
        candidates, negatives = {}, []
        for identifier in sorted(self.images):
            encoded, small = self._image_masks(identifier)
            if encoded:
                candidates[identifier] = (set(encoded), small)
            else:
                negatives.append(identifier)

        def tie(identifier):
            payload = f"hazard_prompt_v1:coco:0:{identifier}".encode("utf-8")
            return hashlib.sha256(payload).hexdigest(), identifier

        selected, missing_strata = [], []
        remaining = dict(candidates)
        negative_pool = sorted(negatives, key=tie)
        for split, budget in budgets.items():
            available_positive = {concept for present, _ in remaining.values() for concept in present}
            available_small = {concept for _, small in remaining.values() for concept in small}
            desired = {concept: ("small" if concept in available_small else "positive")
                       for concept in self.concepts}
            uncovered = set(self.concepts)
            positive_counts = {concept: 0 for concept in self.concepts}
            small_counts = {concept: 0 for concept in self.concepts}

            def priority(identifier):
                present, small = remaining[identifier]
                covered = sum(concept in (small if desired[concept] == "small" else present)
                              for concept in uncovered)
                balance = sum((Fraction(1, 1 + positive_counts[concept])
                               for concept in present), Fraction(0))
                small_balance = sum((Fraction(1, 1 + small_counts[concept])
                                     for concept in small), Fraction(0))
                return -covered, -balance, -small_balance, *tie(identifier)

            split_rows = []
            for _ in range(budget["positive"]):
                if not remaining:
                    break
                identifier = min(remaining, key=priority)
                present, small = remaining.pop(identifier)
                row = {"image_id": identifier, "split": split, "positive": True,
                       "present_concepts": [concept for concept in self.concepts if concept in present],
                       "small_concepts": [concept for concept in self.concepts if concept in small]}
                split_rows.append(row)
                for concept in present:
                    positive_counts[concept] += 1
                for concept in small:
                    small_counts[concept] += 1
                uncovered -= {concept for concept in uncovered
                              if concept in (small if desired[concept] == "small" else present)}
            selected.extend(split_rows)
            for concept in self.concepts:
                if not positive_counts[concept]:
                    missing_strata.append({"split": split, "concept": concept, "stratum": "positive",
                                           "reason": ("unavailable" if concept not in available_positive
                                                      else "selection_budget")})
                if not small_counts[concept]:
                    missing_strata.append({"split": split, "concept": concept, "stratum": "small",
                                           "reason": ("unavailable" if concept not in available_small
                                                      else "selection_budget")})
            if len(split_rows) < budget["positive"]:
                missing_strata.append({"split": split, "stratum": "positive_images",
                                       "expected": budget["positive"], "selected": len(split_rows),
                                       "reason": "insufficient_annotated_images"})
            negative_ids = negative_pool[:budget["negative"]]
            negative_pool = negative_pool[budget["negative"]:]
            selected.extend({"image_id": identifier, "split": split, "positive": False,
                             "present_concepts": [], "small_concepts": []}
                            for identifier in negative_ids)
            if len(negative_ids) < budget["negative"]:
                missing_strata.append({"split": split, "stratum": "target_negative_images",
                                       "expected": budget["negative"], "selected": len(negative_ids),
                                       "reason": "insufficient_annotated_images"})
        return {
            "algorithm": "category_and_small_target_strata_then_inverse_exposure_v1",
            "seed": 0,
            "tie_break": "sha256(hazard_prompt_v1:coco:0:<numeric_image_id>), numeric_image_id",
            "small_target_definition": "0 < decoded_instance_mask_area < 1024 pixels",
            "category_ids": dict(self.category_ids),
            "budgets": budgets,
            "selected": selected,
            "missing_strata": missing_strata,
        }
