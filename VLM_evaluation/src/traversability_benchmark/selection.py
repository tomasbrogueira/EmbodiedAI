"""Freeze an ID-based deployment sample without consulting reference labels."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


def _read_records(path, id_key):
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = record.get(id_key)
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{path}:{line_number}: missing {id_key}")
            if identifier in records:
                raise ValueError(f"Duplicate {id_key}: {identifier}")
            records[identifier] = record
    return records


def _data_path(data_root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Expected POSIX relative data path: {relative!r}")
    portable = PurePosixPath(relative)
    if portable.is_absolute() or ".." in portable.parts or ":" in relative:
        raise ValueError(f"Unsafe data path: {relative}")
    root = Path(data_root).resolve()
    resolved = root.joinpath(*portable.parts).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Data path escapes configured root: {relative}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing data file: {resolved}")
    return resolved


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(entries, data_root):
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        files = [entry["frame"]["image_path"]] + [r["mask_path"] for r in entry["regions"]]
        for relative in files:
            digest.update(relative.encode())
            digest.update(_hash_file(_data_path(data_root, relative)).encode())
    return digest.hexdigest()


def _balanced_pick(entries, count, seed):
    groups = defaultdict(list)
    for entry in entries:
        frame = entry["frame"]
        groups[(frame["source"], frame["scene_id"], frame["sequence_id"])].append(entry)
    for group in groups.values():
        group.sort(key=lambda e: hashlib.sha256(f"{seed}:{e['frame']['frame_id']}".encode()).hexdigest())
    chosen = []
    for index in range(max((len(g) for g in groups.values()), default=0)):
        for key in sorted(groups):
            if index < len(groups[key]):
                chosen.append(groups[key][index])
                if len(chosen) == count:
                    return chosen
    raise ValueError(f"Need {count} eligible frames; only {len(chosen)} available. Prepare real RGB and planning-relevant masks; fixtures cannot fill real coverage.")


def freeze_selection(run_dir, data_root, *, seed=0, fixture=False):
    """Freeze and validate five development/20 test frames, preserving saved IDs."""
    run_dir = Path(run_dir)
    frames = _read_records(run_dir / "frames.jsonl", "frame_id")
    regions = _read_records(run_dir / "regions.jsonl", "region_id")
    by_frame = defaultdict(list)
    split_by_recording = {}
    for frame in frames.values():
        for key in ("source", "scene_id", "sequence_id", "image_path", "split"):
            if not isinstance(frame.get(key), str) or not frame[key]:
                raise ValueError(f"Frame {frame['frame_id']} missing {key}")
        if frame["split"] not in ("development", "test"):
            raise ValueError(f"Invalid split for {frame['frame_id']}")
        if not fixture and (frame.get("fixture") or frame["source"].lower().startswith(("fixture", "synthetic", "fake"))):
            raise ValueError("Fixture frames require fixture=True and cannot count as real results")
        group = (frame["source"], frame["scene_id"], frame["sequence_id"])
        previous = split_by_recording.setdefault(group, frame["split"])
        if previous != frame["split"]:
            raise ValueError(f"Recording split leakage: {group}")
    for region in regions.values():
        if region.get("frame_id") not in frames:
            raise ValueError(f"Region {region['region_id']} has unknown frame_id")
        if not isinstance(region.get("planning_relevant"), bool):
            raise ValueError(f"Region {region['region_id']} needs boolean planning_relevant")
        if region["planning_relevant"]:
            by_frame[region["frame_id"]].append(region)
    entries = {identifier: {"frame": frame, "regions": sorted(by_frame[identifier], key=lambda r: r["region_id"])} for identifier, frame in frames.items() if by_frame[identifier]}
    selection_path = run_dir / "benchmark" / "frozen_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("fixture") != fixture or selection.get("seed") != seed:
            raise ValueError("Saved selection fixture/seed configuration differs; use a separate run")
        refreshed = []
        for key, expected_count, split in (("warmup_frames", 5, "development"), ("test_frames", 20, "test")):
            saved = selection.get(key, [])
            if len(saved) != expected_count:
                raise ValueError(f"Saved selection needs {expected_count} {key}")
            for entry in saved:
                identifier = entry["frame"]["frame_id"]
                current = entries.get(identifier)
                if current != entry or current["frame"]["split"] != split:
                    raise ValueError(f"Frozen selected input changed or missing: {identifier}")
                refreshed.append(current)
        if len({e["frame"]["frame_id"] for e in refreshed}) != 25:
            raise ValueError("Saved selection contains duplicate frame IDs")
        if _fingerprint(refreshed, data_root) != selection.get("fingerprint"):
            raise ValueError("Frozen selected image/mask contents changed")
        return selection
    warmup = _balanced_pick([e for e in entries.values() if e["frame"]["split"] == "development"], 5, seed)
    test = _balanced_pick([e for e in entries.values() if e["frame"]["split"] == "test"], 20, seed)
    selection = {
        "schema_version": 1, "seed": seed, "fixture": fixture,
        "selection_rule": "seeded SHA256 order, round-robin source/scene/recording",
        "warmup_frames": warmup, "test_frames": test,
        "fingerprint": _fingerprint(warmup + test, data_root),
        "coverage": {
            "available_frames_by_split": dict(Counter(f["split"] for f in frames.values())),
            "eligible_frames_by_split": dict(Counter(e["frame"]["split"] for e in entries.values())),
            "selected_test_by_source": dict(Counter(e["frame"]["source"] for e in test)),
            "phone_hazard_clips": "optional_pending" if not any(f["source"].lower().startswith("phone") for f in frames.values()) else "present_unverified",
            "reference_scope": "semantic_material_policy; geometry and cup/cable coverage are not established",
        },
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("x", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return selection
