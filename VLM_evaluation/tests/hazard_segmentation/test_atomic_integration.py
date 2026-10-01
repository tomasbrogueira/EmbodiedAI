"""A transient Windows sharing denial must not corrupt a committed index."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from traversability_hazard_segmentation import common


class AtomicPublication(unittest.TestCase):
    def test_transient_sharing_lock_retains_old_bytes_until_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.json"
            path.write_bytes(b"old committed bytes")
            publish = common._legacy_atomic_bytes
            attempts = []

            def locked_then_publish(destination, content):
                attempts.append(1)
                if len(attempts) <= 2:
                    self.assertEqual(path.read_bytes(), b"old committed bytes")
                    error = PermissionError("Synthetic Windows sharing denial")
                    error.winerror = 32
                    raise error
                return publish(destination, content)

            with patch.object(common, "_legacy_atomic_bytes", side_effect=locked_then_publish), \
                    patch.object(common.time, "sleep"):
                common.atomic_json(path, {"committed": True})
            self.assertEqual(common.read_json(path), {"committed": True})
            self.assertEqual(len(attempts), 3)

    def test_permanent_access_denial_is_reported_and_retains_existing_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.json"
            path.write_bytes(b"existing user artifact")
            error = PermissionError("Synthetic permanent denial")
            error.winerror = 5
            with patch.object(common, "_legacy_atomic_bytes", side_effect=error) as publish, \
                    patch.object(common.time, "sleep"), self.assertRaises(PermissionError):
                common.atomic_json(path, {"committed": True})
            self.assertEqual(path.read_bytes(), b"existing user artifact")
            self.assertEqual(publish.call_count, 5)


if __name__ == "__main__":
    unittest.main()
