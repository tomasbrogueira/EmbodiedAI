"""Saved deployment-profile accounting checks; all inputs are synthetic CPU fixtures."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

TEST_DIRECTORY = Path(__file__).resolve().parent
COMPONENT = TEST_DIRECTORY.parents[1]
sys.path.insert(0, str(COMPONENT / "src"))
sys.path.insert(0, str(TEST_DIRECTORY))

from fixture_inputs import HazardFixture
from traversability_hazard_evaluation import evaluate


class SavedDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = HazardFixture(Path(self.temporary.name))
        self.fixture.add_segmentation_case()

    def evaluate(self):
        self.fixture.save()
        return evaluate(self.fixture.config)

    def assert_blocked(self, report):
        self.assertTrue(report["validation"]["errors"])
        self.assertFalse(report["deployment"]["complete"])
        self.assertFalse(report["comparison"]["model_selection_ready"])
        self.assertIsNone(report["comparison"]["winner"])

    def test_independent_stage_costs_preserve_warm_loading_cold_and_peaks_without_sum(self):
        self.fixture.add_profile("vlm", peak_bytes=100)
        self.fixture.add_profile("sam", elapsed=(2.0, 4.0), peak_bytes=200)
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        rows = report["deployment"]["rows"]
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["component"] for row in rows}, {"vlm", "sam"})
        self.assertTrue(all(row["result_kind"] == "independent_stage_profile" for row in rows))
        vlm = next(row for row in rows if row["component"] == "vlm")
        sam = next(row for row in rows if row["component"] == "sam")
        self.assertEqual(vlm["warm_per_image"]["latency"], {"median_s": 2.0, "p95_s": 2.9})
        self.assertEqual(vlm["loading"], {"elapsed_s": 7.0})
        self.assertEqual(vlm["cold_start"], {"elapsed_s": 5.0})
        self.assertEqual(vlm["warm_per_image"]["memory"]["allocated_peak_bytes"], 100)
        self.assertEqual(sam["memory"]["allocated_peak_bytes"], 200)
        self.assertEqual(sam["sam_query_count"], 4)
        self.assertTrue(all(row["budget"] is None for row in rows))
        self.assertTrue(report["deployment"]["complete"])
        self.assertFalse(report["comparison"]["model_selection_ready"])

    def test_combined_numbers_require_and_retain_their_own_saved_profile(self):
        self.fixture.add_profile("vlm", peak_bytes=100)
        self.fixture.add_profile("sam", peak_bytes=200)
        self.fixture.add_profile("combined", elapsed=(1.5, 2.5), peak_bytes=250)
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        combined = [row for row in report["deployment"]["rows"] if row["component"] == "combined"]
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["result_kind"], "actual_combined_profile")
        self.assertEqual(combined[0]["latency"], {"median_s": 2.0, "p95_s": 2.45})
        self.assertEqual(combined[0]["memory"]["allocated_peak_bytes"], 250)

    def test_missing_repeats_duplicate_frame_repeat_and_summary_count_mismatch_block(self):
        for variant in ("missing", "duplicate", "summary"):
            with self.subTest(variant=variant):
                self.fixture = HazardFixture(Path(self.temporary.name) / variant)
                self.fixture.add_segmentation_case()
                profile = self.fixture.add_profile("sam")
                if variant == "missing":
                    profile["samples"].pop()
                elif variant == "duplicate":
                    duplicate = copy.deepcopy(profile["samples"][0])
                    duplicate["sample_id"] = "different-id-same-frame-repeat"
                    profile["samples"][1] = duplicate
                else:
                    profile["summary"]["measured_frames"] = 1
                self.assert_blocked(self.evaluate())

    def test_profile_fixture_identity_settings_and_selection_mismatches_block(self):
        variants = {"fixture": False, "condition_key": "fixed_policy", "selection_hash": "0" * 64,
                    "model_key": "different-model", "settings": {}}
        for field, value in variants.items():
            with self.subTest(field=field):
                self.fixture = HazardFixture(Path(self.temporary.name) / field)
                self.fixture.add_segmentation_case()
                profile = self.fixture.add_profile("vlm")
                profile["metadata"][field] = value
                self.assert_blocked(self.evaluate())

    def test_synthetic_fixture_cannot_claim_comparison_ready(self):
        profile = self.fixture.add_profile("vlm")
        profile["summary"]["comparison_ready"] = True
        self.assert_blocked(self.evaluate())

    def test_explicit_missing_or_malformed_samples_cannot_be_complete(self):
        for variant in ("missing", "malformed"):
            with self.subTest(variant=variant):
                self.fixture = HazardFixture(Path(self.temporary.name) / variant)
                self.fixture.add_segmentation_case()
                self.fixture.add_profile("sam")
                self.fixture.save()
                path = self.fixture.run / self.fixture.config["deployment"]["profiles"][0]["samples_path"]
                if variant == "missing":
                    path.unlink()
                else:
                    with path.open("a", encoding="utf-8") as stream:
                        stream.write('{"broken":\n')
                report = evaluate(self.fixture.config)
                self.assert_blocked(report)
                self.assertFalse(report["deployment"]["rows"][0]["valid"])

    def test_fixture_resource_fields_do_not_become_real_measurement_readiness(self):
        self.fixture.add_profile("combined", peak_bytes=250)
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        row = report["deployment"]["rows"][0]
        self.assertFalse(row["latency_measured"])
        self.assertFalse(row["resource_measurements_available"])
        self.assertFalse(report["comparison"]["model_selection_ready"])


if __name__ == "__main__":
    unittest.main()
