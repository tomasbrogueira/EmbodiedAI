"""CPU source-import fixtures; these are not real benchmark frames."""

from fractions import Fraction
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from traversability_data.sources import import_phone_video, import_rellis, import_tum, uniform_indices
from traversability_data.storage import read_json


def image(path, value=4, size=(12, 8), semantic=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (size[1], size[0]) if semantic else (size[1], size[0], 3)
    Image.fromarray(np.full(shape, value, dtype=np.uint8)).save(path)
    return path


def recording(root, sequence="00000", size=20, *, semantic=True):
    for index in range(size):
        image(root / sequence / "pylon_camera_node" / f"frame{index}.png", index)
        if semantic:
            image(root / sequence / "pylon_camera_node_label_id" / f"frame{index}.png", 1, semantic=True)


def test_uniform_indices_are_unique_and_cover_endpoints():
    assert uniform_indices(11, 5) == [0, 3, 5, 8, 10]
    assert uniform_indices(2, 1) == [0]
    assert uniform_indices(20, 20) == list(range(20))
    with pytest.raises(ValueError, match="available"):
        uniform_indices(3, 4)


def test_rellis_default_recipe_and_missing_are_honest(tmp_path):
    source, data = tmp_path / "original", tmp_path / "data"
    recording(source)
    result = import_rellis(data, source)
    assert len(result["frames"]) == 20
    assert len(result["semantic_aids"]) == 20
    assert {row["sequence_id"] for row in result["missing"]} == {"00001", "00002", "00003", "00004"}
    assert all(row["split"] == "development" and row["timestamp_s"] is None for row in result["frames"])
    assert (source / "00000/pylon_camera_node/frame0.png").is_file()
    frame = result["frames"][0]
    assert (data / frame["image_path"]).read_bytes() == (source / "00000/pylon_camera_node/frame0.png").read_bytes()


def test_rellis_uniform_natural_order_and_legacy_recipe(tmp_path):
    source, data = tmp_path / "original", tmp_path / "data"
    recording(source, size=31)
    recipe = {"proposed_development": [{"recording": "00000", "keyframes": 3}], "proposed_test": []}
    result = import_rellis(data, source, recipe)
    assert [row["frame_id"] for row in result["frames"]] == ["rellis:00000:frame0", "rellis:00000:frame15", "rellis:00000:frame30"]
    assert len(import_rellis(tmp_path / "legacy", source, "legacy_phone")["frames"]) == 10


def test_rellis_semantic_alignment_duplicate_and_strict_fail_before_publish(tmp_path):
    source, data = tmp_path / "original", tmp_path / "data"
    recording(source, size=2)
    recipe = {"00000": {"keyframes": 2, "split": "development"}}
    image(source / "00000/pylon_camera_node_label_id/frame1.png", 1, (3, 4), semantic=True)
    with pytest.raises(ValueError, match="aligned"):
        import_rellis(data, source, recipe)
    assert not (data / "rellis").exists()
    image(source / "00000/pylon_camera_node_label_id/frame1.png", 1, semantic=True)
    image(source / "00000/pylon_camera_node/frame0.jpg")
    with pytest.raises(ValueError, match="Duplicate"):
        import_rellis(data, source, recipe)


def test_rellis_resume_retains_equal_files_and_rejects_changed_original(tmp_path):
    source, data = tmp_path / "original", tmp_path / "data"
    recording(source, size=2)
    recipe = {"00000": {"keyframes": 2, "split": "development"}}
    result = import_rellis(data, source, recipe)
    prepared = data / result["frames"][0]["image_path"]
    modified = prepared.stat().st_mtime_ns
    assert import_rellis(data, source, recipe) == result
    assert prepared.stat().st_mtime_ns == modified
    image(source / "00000/pylon_camera_node/frame0.png", 88)
    with pytest.raises(FileExistsError, match="differs"):
        import_rellis(data, source, recipe)
    assert prepared.stat().st_mtime_ns == modified


def test_rellis_rgb_only_is_useful_and_aid_coverage_pending(tmp_path):
    source = tmp_path / "original"
    recording(source, size=2, semantic=False)
    recipe = {"00000": {"keyframes": 2, "split": "development"}}
    result = import_rellis(tmp_path / "data", source, recipe)
    assert len(result["frames"]) == 2
    assert not result["semantic_aids"]
    assert result["missing"][0]["reason"].startswith("semantic_ids_absent")
    with pytest.raises(ValueError, match="Incomplete"):
        import_rellis(tmp_path / "strict", source, recipe, allow_partial=False)
    assert not (tmp_path / "strict/rellis").exists()


def test_rellis_recipe_rejects_split_leakage(tmp_path):
    with pytest.raises(ValueError, match="multiple splits"):
        import_rellis(tmp_path, recipe={"proposed_development": [{"recording": "00000", "keyframes": 1}], "proposed_test": [{"recording": "00000", "keyframes": 1}]})


def tum_sequence(source, rows="1305031102.175304 rgb/1305031102.175304.png\n1305031102.211214 rgb/1305031102.211214.png\n"):
    root = source / "rgbd_dataset_freiburg1_floor"
    root.mkdir(parents=True)
    (root / "rgb.txt").write_text("# timestamp filename\n" + rows, encoding="utf-8")
    for stem in ("1305031102.175304", "1305031102.211214"):
        image(root / "rgb" / f"{stem}.png")
    return root


def test_tum_rgb_manifest_preserves_timestamp_and_bytes(tmp_path):
    original = tum_sequence(tmp_path / "original")
    data = tmp_path / "data"
    frames = import_tum(data, original.parent, "freiburg1_floor")
    assert [row["timestamp_s"] for row in frames] == [1305031102.175304, 1305031102.211214]
    assert all(row["source"] == "tum" and row["split"] == "test" for row in frames)
    assert (data / frames[0]["image_path"]).read_bytes() == (original / "rgb/1305031102.175304.png").read_bytes()
    assert import_tum(data, original, "freiburg1_floor") == frames


@pytest.mark.parametrize("rows,pattern", [
    ("1305031102.175304 rgb/1305031102.175304.png\n1305031102.175304 rgb/1305031102.175304.png\n", "Duplicate"),
    ("nan rgb/1305031102.175304.png\n", "timestamp"),
    ("1305031102.175304 ../escape.png\n", "path|relative|escape|traversal"),
    ("1305031102.211214 rgb/1305031102.211214.png\n1305031102.175304 rgb/1305031102.175304.png\n", "chronological"),
])
def test_tum_invalid_manifest_never_publishes(tmp_path, rows, pattern):
    original = tum_sequence(tmp_path / "original", rows)
    data = tmp_path / "data"
    with pytest.raises((ValueError, FileNotFoundError), match=pattern):
        import_tum(data, original, "freiburg1_floor")
    assert not (data / "tum").exists()


class FakeFrame:
    width, height, rotation, is_corrupt = 12, 8, 90, False
    def __init__(self, index, missing=False):
        self.index = index
        self.pts = None if missing else index * index
        self.time_base = Fraction(1, 30)
    def to_image(self):
        return Image.fromarray(np.full((self.height, self.width, 3), self.index, dtype=np.uint8))


class FakeContainer:
    def __init__(self, frames):
        self.frames = frames
        self.streams = SimpleNamespace(video=[SimpleNamespace(metadata={"rotate": "90"})])
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def decode(self, stream):
        return iter(self.frames)


def test_phone_pts_raster_provenance_and_resume(tmp_path, monkeypatch):
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"mock video, not real footage")
    calls = []
    frames = [FakeFrame(index) for index in range(9)]
    def open_video(path):
        calls.append(path)
        return FakeContainer(frames)
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=open_video, __version__="fixture-decoder", library_versions={"fixture": (1, 2, 3)}))
    data = tmp_path / "data"
    result = import_phone_video(data, video, "B", "fixture", keyframes=3)
    assert [row["timestamp_s"] for row in result] == [0.0, 16 / 30, 64 / 30]
    assert all(row["split"] == "test" and row["scene_id"] == "phone:B" for row in result)
    with Image.open(data / result[1]["image_path"]) as raster:
        assert raster.size == (12, 8)
        assert np.asarray(raster)[0, 0, 0] == 4
    metadata = read_json(data / "phone/B/fixture/import_metadata.json")
    assert metadata["orientation_transform_applied"] == "none"
    assert metadata["selected_frame_provenance"][1]["pts"] == 16
    assert metadata["selected_frame_provenance"][1]["time_base"] == "1/30"
    assert metadata["selected_frame_provenance"][1]["display_rotation_degrees"] == 90
    assert len(calls) == 2
    assert import_phone_video(data, video, "B", "fixture", keyframes=3) == result
    assert len(calls) == 2
    video.write_bytes(b"different fixture")
    with pytest.raises(FileExistsError, match="different"):
        import_phone_video(data, video, "B", "fixture", keyframes=3)


def test_phone_missing_pts_is_null_and_no_phone_data_is_pending(tmp_path, monkeypatch):
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: FakeContainer([FakeFrame(0, missing=True)]), __version__="fixture"))
    result = import_phone_video(tmp_path / "data", video, "A", "fixture", keyframes=1)
    assert result[0]["timestamp_s"] is None
    assert result[0]["split"] == "development"
    with pytest.raises(FileNotFoundError, match="pending"):
        import_phone_video(tmp_path / "data", tmp_path / "missing.mp4", "A", "missing")
