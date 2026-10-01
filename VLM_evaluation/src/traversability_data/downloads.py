"""Explicit single-resource downloads and selective, create-only archive imports."""

from __future__ import annotations

import json
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Mapping

from .storage import data_path, exclusive_lock, hash_file, publish_file


class ResourceAccessError(RuntimeError):
    """A public resource could not be obtained or was not an archive."""


def resource_catalog(manifest_path=None) -> dict:
    """Read the exact project's public resources; this never accesses the network."""
    path = Path(manifest_path) if manifest_path else Path(__file__).resolve().parents[2] / "research/traversability/data_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sources = {row["name"]: row for row in manifest["sources"]}
    rellis = sources["RELLIS-3D"]
    resources = {}
    for kind, key, filename in (
        ("RGB images", "rellis_rgb", "Rellis_3D_pylon_camera_node.zip"),
        ("Semantic ID masks", "rellis_semantic", "Rellis_3D_pylon_camera_node_label_id.zip"),
    ):
        row = next(item for item in rellis["resources"] if item["kind"] == kind)
        resources[key] = {**row, "resource_key": key, "filename": filename, "documentation": rellis["documentation"], "transport": "google_drive"}
    for name, key, sequence in (
        ("TUM RGB-D fr1 floor", "tum_floor", "freiburg1_floor"),
        ("TUM RGB-D fr3 walking xyz", "tum_walking_xyz", "freiburg3_walking_xyz"),
    ):
        row = sources[name]
        resources[key] = {
            "resource_key": key, "url": row["url"], "filename": row["url"].rsplit("/", 1)[-1],
            "sequence_id": sequence, "documentation": row["documentation"], "transport": "http",
            "expected_bytes": row.get("archive_bytes_from_http_head"),
        }
    return resources


def _resource(resource) -> dict:
    catalog = resource_catalog()
    key = resource if isinstance(resource, str) else resource.get("resource_key")
    if key not in catalog:
        raise ValueError(f"Unknown resource {key!r}; choose one of {sorted(catalog)}")
    current = catalog[key]
    if isinstance(resource, Mapping) and resource.get("url", current["url"]) != current["url"]:
        raise ValueError("Resource URL differs from the fixed public-source manifest")
    return current


def _validate_archive(path: Path) -> None:
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                if not archive.infolist():
                    raise ValueError("empty ZIP archive")
        else:
            with tarfile.open(path, "r:gz") as archive:
                if archive.next() is None:
                    raise ValueError("empty TGZ archive")
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ResourceAccessError("The response is not a readable dataset archive; HTML quota/login pages are not data") from exc


def download_resource(resource, data_root, *, allow_download=False) -> Path:
    """Download one manifest resource only after explicit opt-in, preserving completed files."""
    row = _resource(resource)
    if allow_download is not True:
        raise PermissionError("Downloads are opt-in: call with allow_download=True for one chosen resource")
    destination = data_path(data_root, f"archives/{row['filename']}", must_exist=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(destination.with_name(destination.name + ".download.lock")):
        if destination.exists():
            _validate_archive(destination)
            return destination
        with tempfile.TemporaryDirectory(prefix=".download-", dir=destination.parent) as temporary:
            partial = Path(temporary) / row["filename"]
            try:
                if row["transport"] == "google_drive":
                    try:
                        import gdown
                    except ImportError as exc:
                        raise ResourceAccessError("Install optional gdown==6.4.1 in your isolated data environment, or download the exact Drive link in a browser and import its archive") from exc
                    file_id = row["url"].split("/d/", 1)[1].split("/", 1)[0]
                    result = gdown.download(id=file_id, output=str(partial), quiet=False, use_cookies=False, resume=False, timeout=30, retries=0)
                    if result is None or not partial.is_file():
                        raise ResourceAccessError("Google Drive did not return a file")
                else:
                    request = urllib.request.Request(row["url"], headers={"User-Agent": "traversability-data/1"})
                    with urllib.request.urlopen(request, timeout=30) as response:
                        if getattr(response, "status", 200) != 200:
                            raise ResourceAccessError(f"HTTP {response.status}")
                        content_type = response.headers.get("Content-Type", "").lower()
                        if "text/html" in content_type:
                            raise ResourceAccessError("The public link returned an HTML login/error page")
                        with partial.open("xb") as output:
                            shutil.copyfileobj(response, output, length=1024 * 1024)
                    if row.get("expected_bytes") and partial.stat().st_size != row["expected_bytes"]:
                        raise ResourceAccessError("The archive size differs from the manifest's verified HTTP size; inspect the link before importing")
                _validate_archive(partial)
                publish_file(partial, destination)
            except Exception as exc:
                if row["transport"] == "google_drive":
                    advice = "Google Drive may be inaccessible or quota limited. The RELLIS publisher recommends waiting 24 hours, then contacting maskjp@tamu.edu if access still fails. Download the exact link in a browser and import the completed ZIP."
                else:
                    advice = "The TUM link could be inaccessible or the transfer incomplete. Retry this single opt-in action or import a completed archive from the exact manifest link."
                raise ResourceAccessError(f"{advice} Resource: {row['url']}. Detail: {exc}") from exc
    return destination


def _safe_member(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    pieces = normalized.rstrip("/").split("/")
    path = PurePosixPath(normalized)
    if not pieces or path.is_absolute() or any(piece in ("", ".", "..") or ":" in piece or "\x00" in piece for piece in pieces):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path


def _selected(path: PurePosixPath, key: str) -> bool:
    parts = path.parts
    if key.startswith("rellis_"):
        directory = "pylon_camera_node" if key == "rellis_rgb" else "pylon_camera_node_label_id"
        return len(parts) >= 3 and parts[-3] in {f"{index:05d}" for index in range(5)} and parts[-2] == directory and path.suffix.lower() in ({".png", ".jpg", ".jpeg"} if key == "rellis_rgb" else {".png"})
    sequence = "rgbd_dataset_" + resource_catalog()[key]["sequence_id"]
    return (len(parts) >= 2 and parts[-2] == sequence and parts[-1] == "rgb.txt") or (len(parts) >= 3 and parts[-3] == sequence and parts[-2] == "rgb" and path.suffix.lower() == ".png")


def import_archive(archive, data_root, resource_key) -> Path:
    """Safely stage only the source's RGB/semantic files; existing equal files are retained."""
    row = _resource(resource_key)
    archive = Path(archive).resolve()
    _validate_archive(archive)
    destination = data_path(data_root, f"raw/{row['resource_key']}", must_exist=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(destination.with_name(destination.name + ".lock")):
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as temporary:
            stage = Path(temporary)
            selected = []
            seen = set()

            def stage_file(name, stream):
                relative = _safe_member(name)
                normalized = relative.as_posix().casefold()
                if normalized in seen:
                    raise ValueError(f"Duplicate archive member: {name}")
                seen.add(normalized)
                if not _selected(relative, row["resource_key"]):
                    return
                target = stage.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
                selected.append((target, relative.as_posix()))

            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as opened:
                    for member in opened.infolist():
                        _safe_member(member.filename)
                        mode = stat.S_IFMT(member.external_attr >> 16)
                        if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                            raise ValueError(f"Links and special archive members are forbidden: {member.filename}")
                        if not member.is_dir():
                            with opened.open(member) as stream:
                                stage_file(member.filename, stream)
            else:
                with tarfile.open(archive, "r:gz") as opened:
                    for member in opened:
                        _safe_member(member.name)
                        if not (member.isfile() or member.isdir()):
                            raise ValueError(f"Links and special archive members are forbidden: {member.name}")
                        if member.isfile():
                            with opened.extractfile(member) as stream:
                                stage_file(member.name, stream)
            if not selected:
                raise ValueError(f"Archive contains no expected {row['resource_key']} files; check its official layout")
            # Resolve every destination and conflict before publishing any member.
            targets = [(source, data_path(destination, relative, must_exist=False)) for source, relative in selected]
            for source, target in targets:
                if target.exists() and (not target.is_file() or hash_file(source) != hash_file(target)):
                    raise FileExistsError(f"Completed extracted file differs: {target}")
            for source, target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                publish_file(source, target)
    return destination
