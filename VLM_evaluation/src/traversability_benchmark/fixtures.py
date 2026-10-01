"""Small, explicitly synthetic CPU smoke inputs; never experiment evidence."""

import json
from pathlib import Path
import struct
import zlib

from .selection import _data_path, freeze_selection


def _png(rgb=False):
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))
    channels = 3 if rgb else 1
    row = b"\x00" + bytes([180] * (8 * channels))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2 if rgb else 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(row * 8)) + chunk(b"IEND", b"")


def prepare_fixture(run_dir, data_root):
    """Create a new marked 25-frame CPU fixture and its frozen benchmark sample."""
    run_dir, data_root = Path(run_dir), Path(data_root)
    if run_dir.exists() or data_root.exists():
        raise FileExistsError("Fixture preparation needs new run and data directories; existing work is preserved")
    run_dir.mkdir(parents=True)
    data_root.mkdir(parents=True)
    frames, regions = [], []
    for split, count in (("development", 5), ("test", 20)):
        for index in range(count):
            identifier = f"fixture:{split}:{index:03d}"
            image_path = f"{split}_{index:03d}.png"
            (data_root / image_path).write_bytes(_png(rgb=True))
            frames.append({"frame_id": identifier, "source": "fixture", "scene_id": f"fixture-{split}", "sequence_id": split, "timestamp_s": None, "image_path": image_path, "split": split, "fixture": True})
            for region_index in range(2):
                mask_path = f"{split}_{index:03d}_{region_index}.png"
                (data_root / mask_path).write_bytes(_png())
                regions.append({"region_id": f"{identifier}:{region_index}", "frame_id": identifier, "mask_path": mask_path, "planning_relevant": True, "selected_for_classification": region_index == 0, "fixture": True})
    for name, records in (("frames", frames), ("regions", regions)):
        (run_dir / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    (run_dir / "fixture.json").write_text(json.dumps({"fixture": True, "purpose": "CPU software smoke check; no real-model measurements"}) + "\n", encoding="utf-8")
    return freeze_selection(run_dir, data_root, fixture=True)


class FakeBackend:
    """Exercise file preparation, each region request and JSON parsing on CPU."""

    def __init__(self, model_key, settings):
        self.model_key, self.settings = model_key, dict(settings)
        self.closed = False
        self.metadata = {"fixture": True, "revision": "synthetic", "preprocessing": {"kind": "synthetic_file_reads"}, "quantization": None}

    def predict_frame(self, frame, regions, data_root):
        """Return one explicitly synthetic prediction for each supplied target."""
        if self.closed:
            raise RuntimeError("Fixture backend is closed")
        image = _data_path(data_root, frame["image_path"]).read_bytes()
        if not image:
            raise ValueError("Empty fixture image")
        predictions = []
        for region in regions:
            mask = _data_path(data_root, region["mask_path"]).read_bytes()
            if not mask:
                raise ValueError("Empty fixture mask")
            raw = json.dumps({"label": "unknown", "semantic_class": "synthetic_fixture", "reason": "CPU fixture, not model evidence"})
            predictions.append({**json.loads(raw), "region_id": region["region_id"], "frame_id": frame["frame_id"], "model_key": self.model_key, "raw_response": raw, "status": "ok", "error_code": None})
        return predictions

    def close(self):
        """Release only this fixture backend's state."""
        self.closed = True
