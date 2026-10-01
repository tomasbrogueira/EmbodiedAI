"""Independent fixed-SAM fixtures, including ignored and unscored pixels."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

TEST_DIRECTORY = Path(__file__).resolve().parent
COMPONENT = TEST_DIRECTORY.parents[1]
sys.path.insert(0, str(COMPONENT / "src"))
sys.path.insert(0, str(TEST_DIRECTORY))

from fixture_inputs import HazardFixture, MODEL, TASK, canonical_hash
from traversability_hazard_evaluation import evaluate


class FixedSamEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = HazardFixture(Path(self.temporary.name))
        self.fixture.add_segmentation_case()
        self.condition = f"vlm__{MODEL}"
        self.artifacts = self.fixture.segmentation[self.condition]

    def evaluate(self):
        self.fixture.save()
        return evaluate(self.fixture.config)

    def row(self, report, source="rellis"):
        rows = [row for row in report["segmentation"]["rows"]
                if row["source"] == source and row["condition_key"] == self.condition]
        self.assertEqual(len(rows), 1, report["segmentation"]["rows"])
        return rows[0]

    def assert_rate(self, metric, numerator, denominator):
        self.assertEqual(metric["numerator"], numerator)
        self.assertEqual(metric["denominator"], denominator)
        if denominator:
            self.assertAlmostEqual(metric["rate"], numerator / denominator)
        else:
            self.assertIsNone(metric["rate"])

    def assert_blocked(self, report):
        self.assertTrue(report["validation"]["errors"])
        self.assertFalse(report["segmentation"]["complete"])
        self.assertFalse(report["comparison"]["comparison_ready"])

    def test_scored_union_excludes_unscored_queries_and_intersects_valid_pixels(self):
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        self.assertTrue(report["segmentation"]["complete"])
        row = self.row(report)
        self.assertEqual((row["pixel_tp"], row["pixel_fn"], row["pixel_fp"]), (1, 1, 1))
        self.assert_rate(row["hazard_recall"], 1, 2)
        self.assert_rate(row["precision"], 1, 2)
        self.assert_rate(row["iou"], 1, 3)
        self.assertEqual(row["unscored_queries"], 1)
        audit = report["segmentation"]["frame_audit"][0]
        self.assertEqual(audit["full_union_path"], self.artifacts["frames"][0]["union_mask_path"])
        self.assertTrue((self.fixture.run / audit["full_union_path"]).is_file())

    def test_failed_required_query_partial_masks_are_empty_for_recall(self):
        query = self.artifacts["queries"][0]
        query.update(status="error", error_code="fixture_segmentation_failure")
        frame = self.artifacts["frames"][0]
        frame.update(completed_queries=1, failed_queries=1, status="error",
                     error_code="fixture_query_failure")
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        row = self.row(report)
        self.assertEqual((row["pixel_tp"], row["pixel_fn"], row["pixel_fp"]), (0, 2, 0))
        self.assert_rate(row["hazard_recall"], 0, 2)
        self.assertEqual(row["failed_queries"], 1)
        self.assertTrue(report["segmentation"]["complete"])

    def test_missing_and_duplicate_required_query_records_block_completeness(self):
        for variant in ("missing", "duplicate"):
            with self.subTest(variant=variant):
                self.fixture = HazardFixture(Path(self.temporary.name) / variant)
                self.fixture.add_segmentation_case()
                self.artifacts = self.fixture.segmentation[self.condition]
                if variant == "missing":
                    self.artifacts["queries"].pop(0)
                else:
                    self.artifacts["queries"].append(copy.deepcopy(self.artifacts["queries"][0]))
                report = self.evaluate()
                self.assert_blocked(report)
                self.assertEqual(self.row(report)["pixel_fn"], 2)

    def test_missing_required_frame_does_not_remove_reference_pixels(self):
        self.artifacts["frames"].clear()
        report = self.evaluate()
        self.assert_blocked(report)
        self.assertEqual(self.row(report)["pixel_fn"], 2)

    def test_query_identity_phrase_alias_and_condition_are_validated(self):
        mutations = {
            "query_id": "invalid_index",
            "phrase": "tree",
            "canonical_concept": "water",
            "condition_key": "fixed_policy",
            "task_id": "region_classification",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                self.fixture = HazardFixture(Path(self.temporary.name) / field)
                self.fixture.add_segmentation_case()
                self.artifacts = self.fixture.segmentation[self.condition]
                self.artifacts["queries"][0][field] = value
                self.assert_blocked(self.evaluate())

    def test_segmentation_metadata_identity_settings_and_selection_are_required(self):
        mutations = {"condition_key": "fixed_policy", "selection_hash": "0" * 64,
                     "dataset_fingerprint": "0" * 64, "fixture": False, "sam_settings": {}}
        for field, value in mutations.items():
            with self.subTest(field=field):
                self.fixture = HazardFixture(Path(self.temporary.name) / field)
                self.fixture.add_segmentation_case()
                self.artifacts = self.fixture.segmentation[self.condition]
                self.artifacts["metadata"][field] = value
                self.assert_blocked(self.evaluate())

    def test_binary_original_size_masks_paths_and_instances_are_validated(self):
        for variant in ("dimension", "nonbinary", "missing", "path_escape", "scores", "union"):
            with self.subTest(variant=variant):
                self.fixture = HazardFixture(Path(self.temporary.name) / variant)
                self.fixture.add_segmentation_case()
                self.artifacts = self.fixture.segmentation[self.condition]
                query = self.artifacts["queries"][0]
                path = self.fixture.run / query["union_mask_path"]
                if variant == "dimension":
                    Image.new("L", (64, 32), 255).save(path)
                elif variant == "nonbinary":
                    Image.new("L", (64, 64), 33).save(path)
                elif variant == "missing":
                    path.unlink()
                elif variant == "path_escape":
                    query["union_mask_path"] = "../outside.png"
                elif variant == "scores":
                    query["scores"] = []
                else:
                    query["union_mask_path"] = self.fixture.mask(
                        f"segmentation/{self.condition}/masks/wrong_query_union.png", [], root=self.fixture.run)
                self.assert_blocked(self.evaluate())

    def test_successful_empty_upstream_prompt_list_requires_valid_empty_frame_mask(self):
        self.fixture.predictions[0].update(prompts=[], raw_response='{"prompts":[]}')
        frame = self.artifacts["frames"][0]
        frame.update(query_ids=[], requested_queries=0, completed_queries=0, failed_queries=0,
                     sam_query_count=0, union_mask_path=self.fixture.mask(
                         f"segmentation/{self.condition}/masks/success_empty.png", [], root=self.fixture.run))
        self.artifacts["queries"].clear()
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        self.assertTrue(report["segmentation"]["complete"])
        row = self.row(report)
        self.assertEqual((row["pixel_tp"], row["pixel_fn"], row["pixel_fp"]), (0, 2, 0))
        self.assert_rate(row["precision"], 0, 0)

    def test_per_concept_coverage_distinguishes_omitted_prompt_from_query_failure(self):
        reference = self.fixture.references[0]
        reference["present_concepts"].append("tree")
        reference["concept_masks"]["tree"] = self.fixture.mask("hazard_references/extra_tree.png", [(8, 20)])
        reference["concept_pixel_counts"]["tree"] = 1
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        concepts = {row["concept"]: row for row in report["segmentation"]["per_concept"]}
        self.assert_rate(concepts["tree"]["prompt_coverage"], 0, 1)
        self.assert_rate(concepts["person"]["prompt_coverage"], 1, 1)
        self.assertEqual(concepts["tree"]["pixel_fn"], 1)
        self.assertEqual(concepts["person"]["pixel_fn"], 1)
        self.artifacts["queries"][0].update(status="error", error_code="fixture_failure")
        self.artifacts["frames"][0].update(completed_queries=1, failed_queries=1, status="error", error_code="fixture_failure")
        failure_report = self.evaluate()
        failure_person = next(row for row in failure_report["segmentation"]["per_concept"] if row["concept"] == "person")
        self.assert_rate(failure_person["prompt_coverage"], 1, 1)
        self.assert_rate(failure_person["successful_query_coverage"], 0, 1)
        self.assertEqual(failure_person["failed_query_present_frames"], 1)

    def add_condition(self, condition, phrases):
        """Handwrite baseline artifacts without a producer or image-specific rewriting."""
        metadata = copy.deepcopy(self.artifacts["metadata"])
        metadata["condition_key"] = condition
        frame_id = self.fixture.frames[0]["frame_id"]
        queries = []
        for index, phrase in enumerate(phrases):
            positions = [(8, 8), (8, 10), (8, 11)] if phrase == "person" else []
            path = self.fixture.mask(f"segmentation/{condition}/masks/fixture_{index:03d}.png",
                                     positions, root=self.fixture.run)
            queries.append({
                "task_id": TASK, "schema_version": 1, "frame_id": frame_id,
                "condition_key": condition, "query_id": f"{condition}:{frame_id}:q{index:03d}",
                "phrase": phrase, "canonical_concept": phrase,
                "mask_paths": [path] if positions else [], "scores": [0.9] if positions else [],
                "returned_instance_count": int(bool(positions)), "union_mask_path": path,
                "status": "ok", "error_code": None,
            })
        union = self.fixture.mask(f"segmentation/{condition}/masks/full_union.png",
                                  [(8, 8), (8, 10), (8, 11)], root=self.fixture.run)
        frame = copy.deepcopy(self.artifacts["frames"][0])
        frame.update(condition_key=condition, query_ids=[query["query_id"] for query in queries],
                     requested_queries=len(phrases), completed_queries=len(phrases), failed_queries=0,
                     sam_query_count=len(phrases), union_mask_path=union)
        added = {"metadata": metadata, "queries": queries, "frames": [frame]}
        self.fixture.segmentation[condition] = added
        self.fixture.config["segmentation"]["conditions"].append(condition)
        return added

    def test_reference_present_and_fixed_policy_rows_have_explicit_roles_and_query_counts(self):
        self.add_condition("reference_present", ["person"])
        self.add_condition("fixed_policy", self.fixture.policy["canonical_prompts"])
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        rows = {row["condition_key"]: row for row in report["segmentation"]["rows"]}
        self.assertEqual(rows[self.condition]["result_kind"], "combined_vlm_sam")
        self.assertEqual(rows["reference_present"]["result_kind"], "segmentation_diagnostic")
        self.assertEqual(rows["fixed_policy"]["result_kind"], "no_vlm_baseline")
        self.assertEqual(rows["reference_present"]["required_queries"], 1)
        self.assertEqual(rows["fixed_policy"]["required_queries"], 16)
        self.assertEqual(rows["fixed_policy"]["unscored_queries"], 4)
        self.assertFalse(report["comparison"]["model_selection_ready"])

    def test_all_conditions_must_use_same_sam_settings(self):
        baseline = self.add_condition("fixed_policy", self.fixture.policy["canonical_prompts"])
        baseline["metadata"]["sam_settings"]["confidence_threshold"] = 0.4
        self.assert_blocked(self.evaluate())

    def pin_selection_inputs(self):
        """Pin current fixture content without borrowing producer hash code."""
        report = self.evaluate()
        self.fixture.selection["observed_input_signature"] = report["identity"]["observed_input_signature"]
        core = {key: value for key, value in self.fixture.selection.items() if key != "selection_hash"}
        digest = canonical_hash(core)
        self.fixture.selection["selection_hash"] = digest
        for condition in self.fixture.segmentation.values():
            condition["metadata"]["selection_hash"] = digest
        self.fixture.config["segmentation"]["selection_hash_recipe"] = "canonical_selection"

    def test_canonical_selection_hash_and_observed_signature_are_independently_verified(self):
        self.pin_selection_inputs()
        verified = self.evaluate()
        self.assertEqual(verified["validation"]["errors"], [])
        selection = verified["segmentation"]["selection"]
        self.assertTrue(selection["hash_verified"])
        self.assertTrue(selection["input_identity_verified"])
        self.assertTrue(selection["identity_verified"])
        self.fixture.selection["sampling"] = {"seed": 999}
        rejected = self.evaluate()
        self.assert_blocked(rejected)
        self.assertTrue(any("canonical hash mismatch" in error["message"] for error in rejected["validation"]["errors"]))

    def test_stale_selection_signature_rejects_same_count_reference_mask_mutation(self):
        self.pin_selection_inputs()
        path = self.fixture.references[0]["concept_masks"]["person"]
        self.fixture.mask(path, [(8, 8), (8, 13)])
        rejected = self.evaluate()
        self.assert_blocked(rejected)
        self.assertTrue(any("observed input signature changed" in error["message"] for error in rejected["validation"]["errors"]))

    def test_extra_malformed_jsonl_line_blocks_condition_with_all_required_rows_present(self):
        self.fixture.save()
        queries = self.fixture.run / "segmentation" / self.condition / "queries.jsonl"
        with queries.open("a", encoding="utf-8") as stream:
            stream.write('{"truncated_json":\n')
        report = evaluate(self.fixture.config)
        self.assert_blocked(report)
        self.assertEqual(self.row(report)["required_queries"], 2)

    def test_missing_reference_mask_keeps_prompt_coverage_but_pixel_scores_are_unavailable(self):
        reference = self.fixture.references[0]
        (self.fixture.data_root / reference["concept_masks"]["person"]).unlink()
        report = self.evaluate()
        self.assert_blocked(report)
        person = next(row for row in report["segmentation"]["per_concept"] if row["concept"] == "person")
        self.assert_rate(person["prompt_coverage"], 1, 1)
        self.assertIsNone(person["pixel_tp"])
        self.assertIsNone(person["hazard_recall"]["rate"])

    def test_upstream_vlm_failure_remains_frame_failure_despite_empty_query_list(self):
        self.fixture.predictions[0].update(prompts=[], raw_response="fixture upstream error",
                                          status="error", error_code="inference_failure")
        self.artifacts["queries"].clear()
        self.artifacts["frames"][0].update(
            query_ids=[], requested_queries=0, completed_queries=0, failed_queries=0,
            sam_query_count=0, upstream_status="error", status="error", error_code="upstream_failure",
            union_mask_path=self.fixture.mask(f"segmentation/{self.condition}/masks/upstream_error.png", [], root=self.fixture.run))
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        row = self.row(report)
        self.assertEqual(row["failed_frames"], 1)
        self.assertEqual(row["pixel_fn"], 2)
        self.assertTrue(report["segmentation"]["complete"])


if __name__ == "__main__":
    unittest.main()
