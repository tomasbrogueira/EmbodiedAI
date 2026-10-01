"""Malformed nested records must remain explicit validation failures."""

from copy import deepcopy
import tempfile
import unittest

from common import config
from traversability_hazard_data import prepare_run, validate_run
from traversability_hazard_data.storage import atomic_json, read_json


class MalformedMetadataTests(unittest.TestCase):
    def test_nested_container_errors_are_actionable(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            run = prepare_run(cfg, fixture=True)
            path = run / "metadata/dataset.json"
            saved = read_json(path)
            for field, bad in (("assets", []), ("policy", None), ("annotations", None),
                               ("reference_details", []), ("selection", None), ("coverage", [])):
                with self.subTest(field=field):
                    mutated = deepcopy(saved)
                    mutated[field] = bad
                    atomic_json(path, mutated)
                    with self.assertRaises(ValueError):
                        validate_run(run, cfg["data_root"])
            atomic_json(path, saved)
            self.assertTrue(validate_run(run, cfg["data_root"])["valid"])


if __name__ == "__main__":
    unittest.main()
