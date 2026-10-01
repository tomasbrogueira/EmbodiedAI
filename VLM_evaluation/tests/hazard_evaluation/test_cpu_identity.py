"""Regression checks for optional declared identity and export preflight."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parents[1] / "src"))
sys.path.insert(0, str(DIRECTORY))

from fixture_inputs import HazardFixture
from traversability_hazard_evaluation import evaluate, export_report


class DeclaredIdentityTests(unittest.TestCase):
    def test_optional_declared_run_and_policy_identity_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            for kind in ("dataset", "model"):
                for name, wrong in (("run_name", "different_run"), ("policy_id", "legacy_policy")):
                    with self.subTest(kind=kind, field=name):
                        fixture = HazardFixture(Path(temporary) / (kind + name))
                        fixture.discovery_case()
                        metadata = fixture.dataset_metadata if kind == "dataset" else fixture.model_metadata
                        metadata[name] = wrong
                        fixture.save()
                        report = evaluate(fixture.config)
                        self.assertTrue(report["validation"]["errors"])
                        self.assertFalse(report["comparison"]["discovery_complete"])
                        self.assertFalse(report["comparison"]["comparison_ready"])

    def test_explicit_optional_identity_binding_requires_field_and_checks_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = HazardFixture(Path(temporary))
            fixture.discovery_case()
            fixture.config["metadata_bindings"] = {"model": {"run_name": "/identity/run"}}
            fixture.save()
            self.assertFalse(evaluate(fixture.config)["comparison"]["discovery_complete"])
            fixture.model_metadata["identity"] = {"run": fixture.config["run_name"]}
            fixture.save()
            self.assertEqual(evaluate(fixture.config)["validation"]["errors"], [])

    def test_named_export_directory_rejected_before_existing_report_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = HazardFixture(Path(temporary))
            fixture.discovery_case()
            fixture.save()
            report = evaluate(fixture.config)
            destination = Path(temporary) / "reports"
            paths = export_report(report, destination)
            original = Path(paths["hazard_report.json"]).read_bytes()
            target = destination / "errors.jsonl"
            target.unlink()
            target.mkdir()
            modified = copy.deepcopy(report)
            modified["limitations"].append("should not be published")
            with self.assertRaises(ValueError):
                export_report(modified, destination)
            self.assertEqual(Path(paths["hazard_report.json"]).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
