"""Explicit opt-in COCO helpers; preparation itself never accesses the network."""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen
import zipfile

from .coco import safe_image_name
from .storage import data_path, exclusive_lock, hash_file, publish_file, read_json, read_jsonl


ANNOTATION_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATION_MEMBER = "annotations/instances_val2017.json"


def _require_opt_in(allow_download):
    if allow_download is not True:
        raise ValueError("Downloads are disabled; explicitly pass allow_download=True to opt in.")


def _official_url(value, *, expected_path):
    if not isinstance(value, str):
        raise ValueError("COCO source has no official coco_url")
    parsed = urlparse(value)
    if (parsed.scheme not in ("http", "https")
            or parsed.hostname != "images.cocodataset.org"
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 80 if parsed.scheme == "http" else 443)
            or parsed.query or parsed.fragment or unquote(parsed.path) != expected_path):
        raise ValueError(f"Expected the official COCO URL for {expected_path}")
    return value


@contextmanager
def _temporary(directory, suffix=".tmp"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".hazard-coco-", suffix=suffix,
                                     delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def _receive(url, path):
    """Stream only an already checked official URL to a private temporary file."""
    request = Request(url, headers={"User-Agent": "hazard_prompt_v1-data/1"})
    with urlopen(request, timeout=60) as response, Path(path).open("wb") as output:
        # Official objects may redirect; do not accept a redirect to another host.
        final = response.geturl() if hasattr(response, "geturl") else url
        parsed = urlparse(final)
        if parsed.hostname != "images.cocodataset.org" or parsed.scheme not in ("http", "https"):
            raise ValueError("COCO download redirected outside the official image host")
        shutil.copyfileobj(response, output, length=1024 * 1024)
        output.flush()
        os.fsync(output.fileno())


def _under_root(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Download destination escapes configured data root: {path}")
    return data_path(root, path.relative_to(root).as_posix(), must_exist=False)


def _lock_path(root, destination, suffix):
    """Preflight reusable lock filenames as strictly as their target assets."""
    root = Path(root).resolve()
    lock = Path(destination).with_name(Path(destination).name + suffix)
    if lock.is_symlink():
        raise ValueError(f"Download lock cannot be a symlink: {lock}")
    return data_path(root, lock.relative_to(root).as_posix(), must_exist=False)


def download_coco_annotations(config, *, allow_download=False):
    """Download the official archive and import only ``instances_val2017.json``.

    No archive is downloaded when configured annotations already exist. Archive
    members are never extracted by filename; traversal and duplicate members are
    rejected, and an existing annotation file is never replaced.
    """
    _require_opt_in(allow_download)
    from .configuration import resolve_config
    cfg = resolve_config(config)
    destination = _under_root(cfg["data_root"], cfg["coco"]["annotation_path"])
    if destination.exists():
        read_json(destination)
        return destination
    archive = data_path(cfg["cache_root"], "coco/annotations_trainval2017.zip", must_exist=False)
    _lock_path(cfg["cache_root"], archive, ".lock")
    _lock_path(cfg["data_root"], destination, ".lock")
    with exclusive_lock(_lock_path(cfg["cache_root"], archive, ".download.lock")):
        if not archive.exists():
            with _temporary(archive.parent, ".zip") as temporary:
                _receive(_official_url(ANNOTATION_URL,
                                      expected_path="/annotations/annotations_trainval2017.zip"), temporary)
                if not zipfile.is_zipfile(temporary):
                    raise ValueError("COCO annotation download is not a ZIP archive")
                publish_file(temporary, archive)
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            names = set()
            for member in members:
                safe_image_name(member.filename.rstrip("/"))
                if member.filename in names:
                    raise ValueError(f"Duplicate COCO archive member: {member.filename}")
                names.add(member.filename)
            matches = [member for member in members if member.filename == ANNOTATION_MEMBER]
            if len(matches) != 1 or matches[0].is_dir():
                raise ValueError("Official archive must contain one annotations/instances_val2017.json")
            with _temporary(destination.parent, ".json") as temporary:
                with zipped.open(matches[0]) as incoming, temporary.open("wb") as output:
                    shutil.copyfileobj(incoming, output, length=1024 * 1024)
                document = read_json(temporary)
                if (not isinstance(document, dict)
                        or any(not isinstance(document.get(key), list)
                               for key in ("images", "annotations", "categories"))):
                    raise ValueError("Downloaded file does not have the COCO instances schema")
                publish_file(temporary, destination)
    return destination


def _check_image(path, image):
    from PIL import Image
    with Image.open(path) as incoming:
        if incoming.size != (image["width"], image["height"]):
            raise ValueError(f"COCO image dimensions differ from annotation: {path}")
        incoming.verify()


def download_selected_coco_images(run_dir, data_root, cache_root=None, *, allow_download=False):
    """Download frozen selected IDs only, from their exact official ``coco_url``.

    Call ``prepare_run`` first with annotations available to freeze selection;
    then call this helper explicitly, and prepare again to fill pending images.
    Completed RGB hashes and files are preserved. This function never selects IDs
    or changes any run artifact.
    """
    _require_opt_in(allow_download)
    run_dir, data_root = Path(run_dir).resolve(), Path(data_root).resolve()
    from .validation import validate_run
    validate_run(run_dir, data_root)
    metadata = read_json(data_path(run_dir, "metadata/dataset.json"))
    if (metadata.get("task_id") != "hazard_prompt_v1"
            or type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 1):
        raise ValueError("COCO image download requires hazard_prompt_v1 metadata")
    if metadata.get("fixture"):
        raise ValueError("Synthetic fixture runs cannot download real COCO images")
    selection = metadata.get("selection", {}).get("coco")
    if not isinstance(selection, dict):
        raise ValueError("COCO annotation selection is pending; import annotations and prepare first")
    image_dir = selection.get("image_dir", "raw/coco/val2017")
    if not isinstance(image_dir, str):
        raise ValueError("Frozen COCO image_dir must be a data-relative POSIX path")
    safe_image_name(image_dir)
    source_images = metadata.get("source_images", {}).get("coco")
    if not isinstance(source_images, dict):
        raise ValueError("Frozen run metadata lacks original COCO image records")
    chosen = selection.get("selected")
    if not isinstance(chosen, list):
        raise ValueError("Frozen run metadata lacks selected COCO IDs")
    identifiers = []
    targets = []
    for row in chosen:
        identifier = row.get("image_id") if isinstance(row, dict) else None
        if type(identifier) is not int or identifier < 0 or identifier in identifiers:
            raise ValueError("Frozen COCO selection has invalid/duplicate image IDs")
        identifiers.append(identifier)
        image = source_images.get(str(identifier))
        if not isinstance(image, dict) or image.get("id") != identifier:
            raise ValueError(f"Missing original COCO image record: {identifier}")
        filename = safe_image_name(image.get("file_name"))
        url = _official_url(image.get("coco_url"), expected_path="/val2017/" + filename)
        target = data_path(data_root, image_dir + "/" + filename, must_exist=False)
        for key in ("width", "height"):
            if type(image.get(key)) is not int or image[key] < 1:
                raise ValueError(f"Invalid frozen COCO image dimensions: {identifier}")
        targets.append((identifier, image, url, target))
    if len({target for _, _, _, target in targets}) != len(targets):
        raise ValueError("Frozen COCO selection has duplicate destination filenames")

    # Validate existing completed artifacts before allowing any network operation.
    existing_hashes, completed_frame_hashes = {}, {}
    for frame in read_jsonl(data_path(run_dir, "frames.jsonl", must_exist=False), missing_ok=True):
        if frame.get("source") == "coco":
            path = data_path(data_root, frame.get("image_path"))
            expected_hash = frame.get("image_sha256")
            if hash_file(path) != expected_hash:
                raise ValueError(f"Completed COCO RGB hash changed: {frame.get('frame_id')}")
            existing_hashes[path] = expected_hash
            completed_frame_hashes[frame["frame_id"]] = expected_hash
    for identifier, image, _, target in targets:
        _lock_path(data_root, target, ".lock")
        _lock_path(data_root, target, ".download.lock")
        if target.exists():
            _check_image(target, image)
            completed_hash = completed_frame_hashes.get(f"coco:val2017:{identifier:012d}")
            expected_hash = completed_hash or existing_hashes.get(target)
            if expected_hash and hash_file(target) != expected_hash:
                raise ValueError(f"Completed COCO source hash changed: {target}")

    cache = Path(cache_root).expanduser().resolve() if cache_root else data_root / ".cache"
    cache = data_path(cache, "coco/selected_images", must_exist=False)
    summary = {"selected_ids": identifiers, "downloaded_ids": [], "existing_ids": []}
    for identifier, image, url, target in targets:
        if target.exists():
            summary["existing_ids"].append(identifier)
            continue
        with exclusive_lock(_lock_path(data_root, target, ".download.lock")):
            if target.exists():
                _check_image(target, image)
                completed_hash = completed_frame_hashes.get(f"coco:val2017:{identifier:012d}")
                if completed_hash and hash_file(target) != completed_hash:
                    raise ValueError(f"Completed COCO source hash changed: {target}")
                summary["existing_ids"].append(identifier)
                continue
            with _temporary(cache, Path(target.name).suffix) as temporary:
                _receive(url, temporary)
                _check_image(temporary, image)
                completed_hash = completed_frame_hashes.get(f"coco:val2017:{identifier:012d}")
                if completed_hash and hash_file(temporary) != completed_hash:
                    raise ValueError(f"Downloaded COCO source differs from completed RGB: {identifier}")
                publish_file(temporary, target)
            summary["downloaded_ids"].append(identifier)
    return summary
