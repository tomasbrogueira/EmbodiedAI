"""Opt-in helper checks use tiny mocked HTTP responses, never real networking."""

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
from PIL import Image

from common import config, sha256, write_json, write_jsonl
from traversability_hazard_data.downloads import (
    download_coco_annotations, download_selected_coco_images,
)


def image_bytes(size=(8, 6), color=(30, 90, 150)):
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, format="JPEG")
    return stream.getvalue()


def annotation_zip(extra_member=None):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("annotations/instances_val2017.json",
                         json.dumps({"images": [], "annotations": [], "categories": []}))
        archive.writestr("annotations/instances_train2017.json", "do not extract this")
        if extra_member:
            archive.writestr(extra_member, "unsafe")
    return stream.getvalue()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.cfg = config(self.base)
        self.data = self.base / "data"
        self.run = self.base / "runs" / "fixture"
        (self.run / "metadata").mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self, *, url=None, image_dir="raw/coco/val2017", fixture=False):
        metadata = {
            "task_id": "hazard_prompt_v1", "schema_version": 1, "fixture": fixture,
            "selection": {"coco": {"image_dir": image_dir,
                                      "selected": [{"image_id": 42, "split": "development", "positive": False}]}},
            "source_images": {"coco": {"42": {"id": 42, "file_name": "000000000042.jpg",
                                                 "width": 8, "height": 6,
                                                 "coco_url": url or "http://images.cocodataset.org/val2017/000000000042.jpg"}}},
        }
        write_json(self.run / "metadata/dataset.json", metadata)
        return metadata

    def test_both_helpers_require_explicit_opt_in_without_network_or_writes(self):
        with mock.patch("traversability_hazard_data.downloads.urlopen", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(ValueError, "disabled"):
                download_coco_annotations(self.cfg)
            with self.assertRaisesRegex(ValueError, "disabled"):
                download_selected_coco_images(self.run, self.data)
        self.assertFalse(self.data.exists())

    def test_archive_imports_only_validation_instances_and_is_idempotent(self):
        with mock.patch("traversability_hazard_data.downloads.urlopen", return_value=BytesIO(annotation_zip())) as network:
            path = download_coco_annotations(self.cfg, allow_download=True)
        self.assertEqual(path, self.data / "raw/coco/annotations/instances_val2017.json")
        self.assertEqual(network.call_count, 1)
        self.assertFalse((path.parent / "instances_train2017.json").exists())
        original = path.stat().st_mtime_ns
        with mock.patch("traversability_hazard_data.downloads.urlopen", side_effect=AssertionError("network")):
            self.assertEqual(download_coco_annotations(self.cfg, allow_download=True), path)
        self.assertEqual(path.stat().st_mtime_ns, original)

    def test_traversal_archive_and_external_destination_are_refused(self):
        with mock.patch("traversability_hazard_data.downloads.urlopen", return_value=BytesIO(annotation_zip("../outside.txt"))):
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                download_coco_annotations(self.cfg, allow_download=True)
        self.assertFalse((self.data / "raw/coco/annotations/instances_val2017.json").exists())
        external = dict(self.cfg, coco={"annotation_path": str(self.base / "outside.json")})
        with mock.patch("traversability_hazard_data.downloads.urlopen", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(ValueError, "escapes"):
                download_coco_annotations(external, allow_download=True)

    def test_symlink_lock_preflight_blocks_network(self):
        # Mock the host check because Windows may disallow creating symlinks.
        with mock.patch("pathlib.Path.is_symlink", return_value=True):
            with mock.patch("traversability_hazard_data.downloads.urlopen", side_effect=AssertionError("network")):
                with self.assertRaisesRegex(ValueError, "lock cannot be a symlink"):
                    download_coco_annotations(self.cfg, allow_download=True)

    def test_only_selected_official_url_is_downloaded_and_existing_is_preserved(self):
        metadata = self.manifest()
        metadata["source_images"]["coco"]["999"] = {
            "id": 999, "file_name": "000000000999.jpg", "width": 8, "height": 6,
            "coco_url": "http://images.cocodataset.org/val2017/000000000999.jpg",
        }
        write_json(self.run / "metadata/dataset.json", metadata)
        with mock.patch("traversability_hazard_data.validation.validate_run") as validation:
            with mock.patch("traversability_hazard_data.downloads.urlopen", return_value=BytesIO(image_bytes())) as network:
                result = download_selected_coco_images(self.run, self.data, self.base / "cache", allow_download=True)
        validation.assert_called_once_with(self.run, self.data)
        self.assertEqual(result["downloaded_ids"], [42])
        self.assertEqual(network.call_count, 1)
        self.assertEqual(network.call_args.args[0].full_url,
                         "http://images.cocodataset.org/val2017/000000000042.jpg")
        target = self.data / "raw/coco/val2017/000000000042.jpg"
        original = target.stat().st_mtime_ns
        with mock.patch("traversability_hazard_data.validation.validate_run"):
            with mock.patch("traversability_hazard_data.downloads.urlopen", side_effect=AssertionError("network")):
                repeat = download_selected_coco_images(self.run, self.data, allow_download=True)
        self.assertEqual(repeat["existing_ids"], [42])
        self.assertEqual(target.stat().st_mtime_ns, original)
        self.assertFalse((target.parent / "000000000999.jpg").exists())

    def test_metadata_validation_and_changed_completed_rgb_block_network(self):
        self.manifest()
        with mock.patch("traversability_hazard_data.validation.validate_run", side_effect=ValueError("fingerprint changed")):
            with mock.patch("traversability_hazard_data.downloads.urlopen", side_effect=AssertionError("network")):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    download_selected_coco_images(self.run, self.data, allow_download=True)
        target = self.data / "raw/coco/val2017/000000000042.jpg"
        target.parent.mkdir(parents=True)
        target.write_bytes(image_bytes())
        write_jsonl(self.run / "frames.jsonl", [{"frame_id": "coco:val2017:42", "source": "coco",
                                                "image_path": target.relative_to(self.data).as_posix(),
                                                "image_sha256": sha256(target)}])
        target.write_bytes(image_bytes((7, 6)))
        with mock.patch("traversability_hazard_data.validation.validate_run"):
            with mock.patch("traversability_hazard_data.downloads.urlopen", side_effect=AssertionError("network")):
                with self.assertRaisesRegex(ValueError, "hash changed"):
                    download_selected_coco_images(self.run, self.data, allow_download=True)

    def test_escaped_paths_unofficial_urls_fixture_and_wrong_dimensions_fail(self):
        for options in ({"image_dir": "../escape"}, {"image_dir": None},
                        {"url": "https://example.org/val2017/000000000042.jpg"},
                        {"url": "http://images.cocodataset.org/train2017/000000000042.jpg"},
                        {"fixture": True}):
            with self.subTest(options=options):
                self.manifest(**options)
                with mock.patch("traversability_hazard_data.validation.validate_run"):
                    with mock.patch("traversability_hazard_data.downloads.urlopen", side_effect=AssertionError("network")):
                        with self.assertRaises(ValueError):
                            download_selected_coco_images(self.run, self.data, allow_download=True)
        self.manifest()
        with mock.patch("traversability_hazard_data.validation.validate_run"):
            with mock.patch("traversability_hazard_data.downloads.urlopen", return_value=BytesIO(image_bytes((7, 6)))):
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    download_selected_coco_images(self.run, self.data, allow_download=True)
        self.assertFalse((self.data / "raw/coco/val2017/000000000042.jpg").exists())

    def test_public_preparation_freezes_missing_rgb_before_opt_in_download_and_fill(self):
        # Six tiny synthetic source-schema inputs exercise real pending/fill
        # mechanics; they are not experimental COCO coverage.
        from test_coco import CATEGORY_IDS, document, encoded
        from traversability_hazard_data import prepare_run, validate_run
        annotation = self.data / "raw/coco/annotations/instances_val2017.json"
        annotation.parent.mkdir(parents=True)
        content = document(images=6, shape=(6, 8))
        one = np.zeros((6, 8), dtype=np.uint8)
        one[1, 1] = 1
        for identifier in range(2):
            content["annotations"].append({"id": identifier, "image_id": identifier,
                                           "category_id": CATEGORY_IDS["person"],
                                           "segmentation": encoded(one), "iscrowd": 0})
        write_json(annotation, content)
        # The hand-built helper manifest lives in another run; use a clean one.
        cfg = dict(self.cfg, run_name="pending_download_integration")
        with mock.patch("traversability_hazard_data.downloads.urlopen", side_effect=AssertionError("implicit network")):
            run = prepare_run(cfg)
        metadata_before = (run / "metadata/dataset.json").read_bytes()
        before = json.loads(metadata_before)
        selected_before = before["selection"]["coco"]["selected"]
        self.assertEqual(validate_run(run, self.data)["frames"], 0)

        def synthetic_response(request, **_):
            identifier = int(Path(request.full_url).stem)
            return BytesIO(image_bytes(color=(20 + 20 * identifier, 90, 150)))

        with mock.patch("traversability_hazard_data.downloads.urlopen", side_effect=synthetic_response) as network:
            downloaded = download_selected_coco_images(run, self.data, self.base / "cache", allow_download=True)
        self.assertEqual(set(downloaded["downloaded_ids"]), {row["image_id"] for row in selected_before})
        self.assertEqual(network.call_count, len(selected_before))
        self.assertEqual((run / "metadata/dataset.json").read_bytes(), metadata_before)
        with mock.patch("traversability_hazard_data.downloads.urlopen", side_effect=AssertionError("implicit network")):
            self.assertEqual(prepare_run(cfg), run)
        after = json.loads((run / "metadata/dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(after["selection"]["coco"]["selected"], selected_before)
        self.assertEqual(validate_run(run, self.data)["frames"], len(selected_before))
        self.assertFalse(after["coverage"]["complete_planned_frame_coverage"])
        # Prepared outputs still validate, but a changed raw-source JPEG of the
        # same dimensions must not be silently accepted by the download helper.
        changed_id = selected_before[0]["image_id"]
        raw_image = self.data / f"raw/coco/val2017/{changed_id:012d}.jpg"
        raw_image.write_bytes(image_bytes(color=(255, 0, 0)))
        with mock.patch("traversability_hazard_data.downloads.urlopen", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(ValueError, "source hash changed"):
                download_selected_coco_images(run, self.data, allow_download=True)
        raw_image.unlink()
        with mock.patch("traversability_hazard_data.downloads.urlopen", return_value=BytesIO(image_bytes(color=(255, 0, 0)))):
            with self.assertRaisesRegex(ValueError, "differs from completed"):
                download_selected_coco_images(run, self.data, allow_download=True)
        self.assertFalse(raw_image.exists())


if __name__ == "__main__":
    unittest.main()
