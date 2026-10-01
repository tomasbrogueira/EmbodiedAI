"""Small synthetic archives and mock transfers, never public dataset downloads."""

import io
from pathlib import Path
import stat
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from traversability_data import downloads


def zipped(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as opened:
        for name, payload in entries.items():
            opened.writestr(name, payload)
    return output.getvalue()


def tarred(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as opened:
        for name, payload in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            opened.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def test_catalog_uses_only_exact_manifest_public_resources():
    catalog = downloads.resource_catalog()
    assert set(catalog) == {"rellis_rgb", "rellis_semantic", "tum_floor", "tum_walking_xyz"}
    assert catalog["rellis_rgb"]["url"] == "https://drive.google.com/file/d/1F3Leu0H_m6aPVpZITragfreO_SGtL2yV/view"
    assert catalog["rellis_semantic"]["url"] == "https://drive.google.com/file/d/16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5/view"
    assert catalog["tum_floor"]["url"].endswith("/freiburg1/rgbd_dataset_freiburg1_floor.tgz")
    assert catalog["tum_walking_xyz"]["url"].endswith("/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz")


def test_download_requires_per_resource_opt_in_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network must remain disabled"))
    with pytest.raises(PermissionError, match="opt-in"):
        downloads.download_resource("tum_floor", tmp_path)
    assert not (tmp_path / "archives").exists()
    with pytest.raises(ValueError, match="differs"):
        downloads.download_resource({"resource_key": "tum_floor", "url": "https://example.invalid"}, tmp_path, allow_download=True)


def test_drive_download_is_lazy_single_file_absolute_temp_and_preserves_complete(tmp_path, monkeypatch):
    payload = zipped({"Rellis-3D/00000/pylon_camera_node/frame0.png": b"CPU fixture"})
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["output"])
        assert path.is_absolute() and path.parent.name.startswith(".download-")
        path.write_bytes(payload)
        return str(path)
    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=download))
    result = downloads.download_resource("rellis_rgb", tmp_path, allow_download=True)
    assert result.read_bytes() == payload
    assert len(calls) == 1
    assert calls[0]["id"] == "1F3Leu0H_m6aPVpZITragfreO_SGtL2yV"
    assert calls[0]["use_cookies"] is False and calls[0]["resume"] is False and calls[0]["retries"] == 0
    assert not list(result.parent.glob(".download-*"))
    original_mtime = result.stat().st_mtime_ns
    assert downloads.download_resource("rellis_rgb", tmp_path, allow_download=True) == result
    assert len(calls) == 1 and result.stat().st_mtime_ns == original_mtime


def test_drive_quota_error_is_actionable_and_leaves_no_completed_file(tmp_path, monkeypatch):
    def denied(**kwargs):
        Path(kwargs["output"]).write_bytes(b"<html>Quota exceeded</html>")
        return kwargs["output"]
    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=denied))
    with pytest.raises(downloads.ResourceAccessError, match="24 hours"):
        downloads.download_resource("rellis_semantic", tmp_path, allow_download=True)
    assert not (tmp_path / "archives/Rellis_3D_pylon_camera_node_label_id.zip").exists()
    assert not list((tmp_path / "archives").glob(".download-*"))


class Response(io.BytesIO):
    status = 200
    def __init__(self, data, content_type="application/gzip"):
        super().__init__(data)
        self.headers = {"Content-Type": content_type}


def test_tum_http_stream_validates_content_and_temp_cleanup(tmp_path, monkeypatch):
    payload = tarred({"rgbd_dataset_freiburg1_floor/rgb.txt": b"# fixture"})
    row = downloads.resource_catalog()["tum_floor"]
    row["expected_bytes"] = len(payload)
    monkeypatch.setattr(downloads, "_resource", lambda resource: row)
    requested = []
    def response(request, timeout):
        requested.append((request.full_url, timeout))
        return Response(payload)
    monkeypatch.setattr(downloads.urllib.request, "urlopen", response)
    result = downloads.download_resource("tum_floor", tmp_path, allow_download=True)
    assert result.read_bytes() == payload
    assert requested == [(row["url"], 30)]
    assert not list(result.parent.glob(".download-*"))


def test_tum_html_or_incomplete_response_never_publishes(tmp_path, monkeypatch):
    row = downloads.resource_catalog()["tum_floor"]
    monkeypatch.setattr(downloads, "_resource", lambda resource: row)
    monkeypatch.setattr(downloads.urllib.request, "urlopen", lambda *args, **kwargs: Response(b"<html>denied</html>", "text/html"))
    with pytest.raises(downloads.ResourceAccessError, match="HTML"):
        downloads.download_resource("tum_floor", tmp_path, allow_download=True)
    assert not (tmp_path / "archives" / row["filename"]).exists()


def test_rellis_archive_selective_import_resumption_and_conflict(tmp_path):
    archive = tmp_path / "fixture.zip"
    archive.write_bytes(zipped({
        "Rellis-3D/00000/pylon_camera_node/frame0.png": b"original fixture bytes",
        "Rellis-3D/00000/pylon_camera_node_label_id/frame0.png": b"separate aid",
        "Rellis-3D/00000/os1_cloud_node_kitti_bin/frame0.bin": b"excluded modality",
    }))
    data = tmp_path / "data"
    result = downloads.import_archive(archive, data, "rellis_rgb")
    prepared = result / "Rellis-3D/00000/pylon_camera_node/frame0.png"
    assert prepared.read_bytes() == b"original fixture bytes"
    assert not (result / "Rellis-3D/00000/pylon_camera_node_label_id").exists()
    assert not (result / "Rellis-3D/00000/os1_cloud_node_kitti_bin").exists()
    mtime = prepared.stat().st_mtime_ns
    assert downloads.import_archive(archive, data, "rellis_rgb") == result
    assert prepared.stat().st_mtime_ns == mtime
    archive.write_bytes(zipped({"Rellis-3D/00000/pylon_camera_node/frame0.png": b"conflicting", "Rellis-3D/00000/pylon_camera_node/frame1.png": b"new"}))
    with pytest.raises(FileExistsError, match="differs"):
        downloads.import_archive(archive, data, "rellis_rgb")
    assert prepared.read_bytes() == b"original fixture bytes"
    assert not prepared.with_name("frame1.png").exists()


def test_tum_archive_excludes_depth_and_trajectory(tmp_path):
    archive = tmp_path / "fixture.tgz"
    prefix = "rgbd_dataset_freiburg3_walking_xyz"
    archive.write_bytes(tarred({f"{prefix}/rgb.txt": b"# fixture", f"{prefix}/rgb/0.png": b"fixture RGB", f"{prefix}/depth/0.png": b"excluded", f"{prefix}/groundtruth.txt": b"excluded"}))
    result = downloads.import_archive(archive, tmp_path / "data", "tum_walking_xyz")
    assert (result / prefix / "rgb/0.png").is_file()
    assert (result / prefix / "rgb.txt").is_file()
    assert not (result / prefix / "depth").exists()
    assert not (result / prefix / "groundtruth.txt").exists()


@pytest.mark.parametrize("name", ["../escape.png", "/absolute.png", "C:/drive.png", "safe/../escape.png", "safe\\..\\escape.png"])
def test_archive_rejects_unsafe_even_unselected_members(tmp_path, name):
    archive = tmp_path / "fixture.zip"
    archive.write_bytes(zipped({"00000/pylon_camera_node/frame0.png": b"fixture", name: b"malicious"}))
    with pytest.raises(ValueError, match="Unsafe"):
        downloads.import_archive(archive, tmp_path / "data", "rellis_rgb")
    assert not (tmp_path / "data/raw/rellis_rgb").exists()


def test_archive_rejects_zip_symlinks_and_duplicate_paths(tmp_path):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as opened:
        member = zipfile.ZipInfo("00000/pylon_camera_node/frame0.png")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        opened.writestr(member, "outside")
    with pytest.raises(ValueError, match="Links"):
        downloads.import_archive(archive, tmp_path / "data", "rellis_rgb")
    with zipfile.ZipFile(archive, "w") as opened:
        opened.writestr("00000/pylon_camera_node/frame0.png", b"one")
        opened.writestr("00000/pylon_camera_node/FRAME0.png", b"two")
    with pytest.raises(ValueError, match="Duplicate"):
        downloads.import_archive(archive, tmp_path / "data", "rellis_rgb")


def test_tar_links_are_forbidden(tmp_path):
    archive = tmp_path / "fixture.tgz"
    with tarfile.open(archive, "w:gz") as opened:
        member = tarfile.TarInfo("rgbd_dataset_freiburg1_floor/rgb/0.png")
        member.type = tarfile.SYMTYPE
        member.linkname = "/outside"
        opened.addfile(member)
    with pytest.raises(ValueError, match="Links"):
        downloads.import_archive(archive, tmp_path / "data", "tum_floor")
