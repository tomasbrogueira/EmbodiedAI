"""Eleven explicitly synthetic CPU samples, isolated from real imports."""

from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

from .storage import atomic_json, data_path, digest, publish_file, read_json


def create_fixture_sources(data_root, policy, *, profile="small"):
    root = data_path(data_root, "fixtures/hazard_prompt_v1/sources", must_exist=False)
    with tempfile.TemporaryDirectory(prefix="hazard-fixture-") as temp:
        temp = Path(temp)
        def png(relative, array):
            staged = temp / "asset.png"
            Image.fromarray(array).save(staged, format="PNG")
            data_path(root, relative + ".lock", must_exist=False)
            publish_file(staged, data_path(root, relative, must_exist=False))
        per_sequence = 20 if profile == "protocol" else 1
        for sample in range(5 * per_sequence):
            i, local = divmod(sample, per_sequence)
            labels = np.ones((40, 50), dtype=np.uint8)
            if i < 3:
                labels.flat[:(80, 100, 120)[i]] = 0
                labels.flat[200] = 6
                labels.flat[201] = 31
                labels.flat[202] = 4 if i == 0 else 17
                labels.flat[203:210] = 7
            elif i == 3:
                labels[:] = 7
            else:
                for j, ids in enumerate(policy["datasets"]["rellis"]["hazard_label_ids"].values()):
                    labels.flat[j] = ids[0]
            rgb = np.zeros((40, 50, 3), dtype=np.uint8)
            rgb[:] = [20 + i * 30, 80 + local, 140]
            name = f"synthetic_{i:03d}.png" if per_sequence == 1 else f"synthetic_{i:03d}_{local:03d}.png"
            png(f"rellis/{i:05d}/pylon_camera_node/{name}", rgb)
            png(f"rellis/{i:05d}/pylon_camera_node_label_id/{name}", labels)
        from pycocotools import mask as coco_mask
        categories = [{"id": 101 + i * 7, "name": name}
                      for i, name in enumerate(policy["datasets"]["coco"]["scored_category_names"])]
        categories.append({"id": 999, "name": "truck"})
        images, annotations = [], []
        count, positives = (40, 32) if profile == "protocol" else (6, 4)
        for i in range(count):
            image_id = 900001 + i
            name = f"synthetic_{image_id}.png"
            rgb = np.zeros((40, 50, 3), dtype=np.uint8)
            rgb[:] = [140, 20 + (i * 30 if profile == "small" else i * 5), 80]
            png(f"coco/val2017/{name}", rgb)
            images.append({"id": image_id, "width": 50, "height": 40, "file_name": name,
                           "coco_url": f"https://images.cocodataset.org/val2017/{name}"})
            target_ids = [c["id"] for c in categories[:-1]] if i < positives else [999]
            for j, category in enumerate(target_ids):
                # Disjoint small polygons plus compressed/uncompressed crowd RLE.
                segmentation = [[2 + j * 7, 2, 5 + j * 7, 2, 5 + j * 7, 5, 2 + j * 7, 5]]
                crowd = 0
                if i == 1 or (i == 2 and j == 0):
                    array = np.zeros((40, 50), dtype=np.uint8, order="F")
                    array[10:12, 2 + j * 7:4 + j * 7] = 1
                    encoded = coco_mask.encode(array)
                    if i == 1:
                        segmentation = {"size": encoded["size"], "counts": encoded["counts"].decode("ascii")}
                    else:
                        flat = array.flatten(order="F")
                        counts, previous, length = [], 0, 0
                        for value in flat:
                            if int(value) == previous:
                                length += 1
                            else:
                                counts.append(length)
                                previous, length = int(value), 1
                        counts.append(length)
                        segmentation = {"size": [40, 50], "counts": counts}
                    crowd = 1
                annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                    "category_id": category, "segmentation": segmentation,
                                    "iscrowd": crowd, "area": 9, "bbox": [2 + j * 7, 2, 3, 3]})
                if i == 2 and j == 0:
                    annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                        "category_id": category, "segmentation": [[2, 2, 5, 2, 5, 5, 2, 5]],
                                        "iscrowd": 0, "area": 9, "bbox": [2, 2, 3, 3]})
        content = {"info": {"description": "EXPLICITLY SYNTHETIC CPU fixture; not COCO experimental data"},
                   "images": images, "categories": categories, "annotations": annotations}
        annotation = data_path(root, "coco/annotations/instances_val2017.json", must_exist=False)
        if annotation.exists():
            if digest(read_json(annotation)) != digest(content):
                raise ValueError("Completed synthetic fixture annotations changed")
        else:
            atomic_json(annotation, content)
    return root / "rellis", annotation, root / "coco/val2017"
