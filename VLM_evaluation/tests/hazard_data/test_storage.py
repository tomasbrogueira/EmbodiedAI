"""Portable paths and atomic storage protect external roots and frozen files."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import file_snapshot
from traversability_hazard_data import storage


class StorageTests(unittest.TestCase):
    def test_posix_and_windows_escape_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for path in ("../escape.png", "/escape.png", "C:/escape.png", "C:\\escape.png",
                         "images\\image.png", "images/../image.png", "images/./image.png",
                         "images//image.png", "", "asset:stream", "image\x00.png"):
                with self.subTest(path=repr(path)), self.assertRaises(ValueError):
                    storage.data_path(temporary, path, must_exist=False)
            self.assertEqual(storage.data_path(temporary, "rgb/image.png", must_exist=False),
                             Path(temporary) / "rgb/image.png")

    def test_symlink_escape_is_rejected_when_host_allows_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "root"
            root.mkdir()
            try:
                (root / "linked").symlink_to(base, target_is_directory=True)
            except OSError:
                self.skipTest("Directory symlinks are not available on this host")
            with self.assertRaises(ValueError):
                storage.data_path(root, "linked/escape.png", must_exist=False)

    def test_atomic_write_failure_preserves_previous_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            storage.atomic_jsonl(path, [{"original": True}])
            original = path.read_bytes()
            # The permitted legacy storage helper owns the actual atomic replace.
            with patch("traversability_data.storage.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    storage.atomic_jsonl(path, [{"replaced": True}])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(set(file_snapshot(temporary)), {"records.jsonl"})


if __name__ == "__main__":
    unittest.main()
