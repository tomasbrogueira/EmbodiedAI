"""Small synthetic CPU demonstration, excluded from real experiment coverage."""

from pathlib import Path
import tempfile

from .storage import data_path, publish_file, read_jsonl
from .run import prepare_run, coverage_summary


def create_cpu_fixture(data_root, run_dir):
    """Demonstrate aligned import, selection and annotation persistence without downloads."""
    import numpy as np
    from PIL import Image
    from .masks import import_sam_masks, freeze_selection
    from .annotations import save_annotation

    data_root, run_dir = Path(data_root), Path(run_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    frames, aids, mask_records = [], {}, []
    with tempfile.TemporaryDirectory(prefix=".cpu-fixture-", dir=data_root) as staging:
        staging = Path(staging)
        for index, split in enumerate(("development", "test")):
            frame_id = f"fixture:{split}:000"
            relative = f"fixture/images/{split}.png"
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            rgb[:, :, 0] = np.arange(32, dtype=np.uint8) * 7
            rgb[:, :, 1] = 100 + index * 40
            rgb[:, :, 2] = np.arange(24, dtype=np.uint8)[:, None] * 8
            image = staging / f"{split}.png"
            Image.fromarray(rgb).save(image)
            publish_file(image, data_path(data_root, relative, must_exist=False))
            frames.append({"frame_id": frame_id, "source": "fixture", "scene_id": f"fixture:{split}",
                           "sequence_id": split, "timestamp_s": None, "image_path": relative, "split": split})
            semantic = staging / f"{split}-semantic.png"
            Image.fromarray(np.ones((24, 32), dtype=np.uint8)).save(semantic)
            aid_relative = f"fixture/semantic/{split}.png"
            publish_file(semantic, data_path(data_root, aid_relative, must_exist=False))
            aids[frame_id] = aid_relative
            for mask_index in range(9):
                mask = np.zeros((24, 32), dtype=np.uint8)
                mask[2:mask_index + 4, 2:mask_index + 6] = 255
                path = staging / f"{split}-mask-{mask_index}.png"
                Image.fromarray(mask).save(path)
                mask_records.append({"region_id": f"{frame_id}:mask{mask_index:02d}",
                                     "frame_id": frame_id, "source_mask_path": str(path)})
        prepare_run(run_dir, frames, data_root,
                    metadata={"is_fixture": True, "recipe": "synthetic_cpu_fixture",
                              "expected_frames": {"development": 20, "test": 80}}, semantic_aids=aids)
        import_sam_masks(run_dir, data_root, mask_records, mask_origin="synthetic_fixture")
    selection = freeze_selection(run_dir, data_root, seed=0)
    demonstration_region = selection["frames"][frames[0]["frame_id"]]["selected_region_ids"][0]
    old = {row["region_id"]: row for row in read_jsonl(run_dir / "annotations.jsonl")}
    if old[demonstration_region]["annotation_status"] != "complete":
        save_annotation(run_dir, demonstration_region, "traversable", mask_quality="valid",
                        semantic_class="synthetic dirt", hazard_type=None)
    return {"run_dir": run_dir, "demonstration_region_id": demonstration_region,
            "selection": selection, "coverage": coverage_summary(run_dir)}
