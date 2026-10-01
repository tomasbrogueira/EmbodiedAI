"""Hand-calculated CPU contracts using independent, explicitly marked fixtures."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

TEST_DIRECTORY = Path(__file__).resolve().parent
COMPONENT = TEST_DIRECTORY.parents[1]
sys.path.insert(0, str(COMPONENT / "src"))
sys.path.insert(0, str(TEST_DIRECTORY))

from fixture_inputs import HazardFixture, MODEL, write_json, write_jsonl
from traversability_hazard_evaluation import evaluate, export_report


class HazardEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = HazardFixture(Path(self.temporary.name))

    def evaluate(self):
        self.fixture.save()
        return evaluate(self.fixture.config)

    def model_row(self, report, source="rellis", split="test"):
        rows = [row for row in report["concepts"]["rows"]
                if row["model_key"] == MODEL and row["source"] == source and row["split"] == split]
        self.assertEqual(len(rows), 1, report["concepts"]["rows"])
        return rows[0]

    def assert_rate(self, value, numerator, denominator):
        self.assertEqual(set(value), {"numerator", "denominator", "rate"})
        self.assertEqual(value["numerator"], numerator)
        self.assertEqual(value["denominator"], denominator)
        if denominator:
            self.assertAlmostEqual(value["rate"], numerator / denominator)
        else:
            self.assertIsNone(value["rate"])

    def assert_blocked(self, report):
        self.assertTrue(report["validation"]["errors"])
        self.assertFalse(report["comparison"]["comparison_ready"])
        self.assertIsNone(report["comparison"]["winner"])

    def test_hand_calculated_alias_sets_and_different_recall_precision_populations(self):
        self.fixture.discovery_case()
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        self.assertTrue(report["fixture"])
        self.assertFalse(report["comparison"]["comparison_ready"])
        row = self.model_row(report)
        self.assert_rate(row["metrics"]["concept_recall"], 2, 4)
        self.assert_rate(row["metrics"]["supported_precision"], 1, 2)
        self.assertEqual(row["recall_tp"], 2)
        self.assertEqual(row["recall_fn"], 2)
        self.assertEqual(row["precision_tp"], 1)
        self.assertEqual(row["scored_fp"], 1)
        self.assert_rate(row["metrics"]["tiny_concept_recall"], 2, 3)
        audits = report["concepts"]["phrase_audit"]
        entries = [item for audit in audits for item in audit["phrases"]]
        raw = [item["raw_phrase"] for item in entries]
        self.assertIn(" PUDDLE ", raw)
        categories = {item["raw_phrase"]: item["category"] for item in entries}
        self.assertEqual(categories["cup"], "source_unscored")
        self.assertEqual(categories["traffic cone"], "unknown_unscored")
        self.assertEqual(categories["hazards"], "vague_unusable")
        self.assertNotIn("hallucination", json.dumps(audits).lower())
        missed = {(item["frame_id"], item["concept"]) for item in report["concepts"]["missed_concepts"]}
        self.assertEqual(missed, {(self.fixture.frames[0]["frame_id"], "tree"),
                                  (self.fixture.frames[1]["frame_id"], "water")})

    def test_each_input_is_joined_by_ids_not_jsonl_position_and_splits_remain_separate(self):
        self.fixture.discovery_case()
        dev = self.fixture.frame(source="coco", split="development", concepts={"cup": [(8, 8)]})
        self.fixture.predict(dev, ["cup"])
        test = self.fixture.frame(source="coco", split="test", concepts={"dog": [(8, 9)]})
        self.fixture.predict(test, [])
        before = self.evaluate()
        for seed, records in enumerate((self.fixture.frames, self.fixture.references, self.fixture.predictions)):
            random.Random(seed + 7).shuffle(records)
        after = self.evaluate()
        self.assertEqual(before["concepts"]["rows"], after["concepts"]["rows"])
        self.assert_rate(self.model_row(after, "coco", "development")["metrics"]["concept_recall"], 1, 1)
        self.assert_rate(self.model_row(after, "coco", "test")["metrics"]["concept_recall"], 0, 1)
        self.assertEqual(after["comparison"]["final_split"], "test")

    def test_valid_empty_and_saved_error_both_miss_positives_only_error_is_failure(self):
        first = self.fixture.frame(concepts={"water": [(10, 10)]})
        self.fixture.predict(first, [])
        second = self.fixture.frame(concepts={"tree": [(10, 20)]})
        self.fixture.predict(second, [], status="error", error_code="invalid_json")
        row = self.model_row(self.evaluate())
        self.assert_rate(row["metrics"]["concept_recall"], 0, 2)
        self.assert_rate(row["metrics"]["supported_precision"], 0, 0)
        self.assertEqual(row["prediction_failures"], 1)
        self.assertEqual(row["successful_empty_lists"], 1)
        self.assert_rate(row["metrics"]["failure_rate"], 1, 2)

    def test_missing_and_duplicate_predictions_block_without_shrinking_positive_denominator(self):
        for variant in ("missing", "duplicate"):
            with self.subTest(variant=variant):
                self.fixture = HazardFixture(Path(self.temporary.name) / variant)
                self.fixture.discovery_case()
                if variant == "missing":
                    self.fixture.predictions.pop()
                else:
                    self.fixture.predictions.append(copy.deepcopy(self.fixture.predictions[-1]))
                report = self.evaluate()
                self.assert_blocked(report)
                row = self.model_row(report)
                self.assert_rate(row["metrics"]["concept_recall"], 1, 4)
                self.assertEqual(row["recall_fn"], 3)
                self.assertFalse(row["predictions_complete"])

    def test_pending_and_missing_references_are_coverage_not_negative_truth(self):
        complete = self.fixture.frame(concepts={"water": [(10, 10)]})
        self.fixture.predict(complete, ["water"])
        pending = self.fixture.frame(status="pending")
        self.fixture.predict(pending, ["tree"])
        missing = self.fixture.frame()
        self.fixture.predict(missing, ["tree"])
        self.fixture.references.pop()
        report = self.evaluate()
        row = self.model_row(report)
        self.assert_rate(row["metrics"]["concept_recall"], 1, 1)
        self.assert_rate(row["metrics"]["supported_precision"], 1, 1)
        self.assertEqual(row["reference_pending"], 1)
        self.assertEqual(row["reference_missing"], 1)
        self.assertFalse(report["comparison"]["discovery_complete"])

    def test_legacy_region_records_are_rejected(self):
        frame = self.fixture.frame(concepts={"water": [(10, 10)]})
        self.fixture.predictions = [{"frame_id": frame["frame_id"], "region_id": "legacy:r000",
                                     "label": "non_traversable", "status": "ok"}]
        report = self.evaluate()
        self.assert_blocked(report)
        self.assert_rate(self.model_row(report)["metrics"]["concept_recall"], 0, 1)

    def test_metadata_identity_and_provenance_mismatches_are_errors(self):
        variants = {"task_id": "region_classification", "schema_version": 2,
                    "policy_hash": "a" * 64, "alias_hash": "b" * 64,
                    "fixture": False, "dataset_fingerprint": "c" * 64,
                    "model_key": "different_model"}
        for field, value in variants.items():
            with self.subTest(field=field):
                self.fixture = HazardFixture(Path(self.temporary.name) / field)
                self.fixture.discovery_case()
                self.fixture.model_metadata[field] = value
                self.assert_blocked(self.evaluate())

    def test_declared_source_split_selected_and_requested_coverage_cannot_hide_missing_frames(self):
        self.fixture.discovery_case()
        self.fixture.dataset_metadata["coverage"]["expected_frames"] = {"rellis": {"test": 3}}
        self.assert_blocked(self.evaluate())
        self.fixture.dataset_metadata["coverage"]["expected_frames"] = {"rellis": {"test": 2}}
        self.fixture.model_metadata["requested_frame_ids"].pop()
        self.assert_blocked(self.evaluate())
        self.fixture.model_metadata["requested_frame_ids"] = [frame["frame_id"] for frame in self.fixture.frames]
        self.fixture.dataset_metadata["selected_frame_ids"].append("rellis:00001:missing")
        self.assert_blocked(self.evaluate())

    def test_escaped_asset_paths_invalid_dimensions_nonbinary_and_hash_mismatches(self):
        for variant in ("escape", "dimension", "nonbinary", "hash", "image_dimension"):
            with self.subTest(variant=variant):
                self.fixture = HazardFixture(Path(self.temporary.name) / variant)
                self.fixture.discovery_case()
                reference = self.fixture.references[0]
                if variant == "escape":
                    reference["concept_masks"]["water"] = "../outside.png"
                elif variant == "dimension":
                    path = self.fixture.data_root / reference["concept_masks"]["water"]
                    Image.new("L", (32, 32), 255).save(path)
                elif variant == "nonbinary":
                    path = self.fixture.data_root / reference["concept_masks"]["water"]
                    Image.new("L", (64, 64), 127).save(path)
                elif variant == "hash":
                    self.fixture.frames[0]["image_sha256"] = "e" * 64
                else:
                    self.fixture.frames[0]["width"] = 63
                self.assert_blocked(self.evaluate())

    def test_exact_normalized_duplicates_and_mapped_alias_duplicates_do_not_inflate_scores(self):
        frame = self.fixture.frame(concepts={"water": [(10, 10)]})
        self.fixture.predict(frame, ["Water", " water  ", "puddle", "standing water"])
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        row = self.model_row(report)
        self.assert_rate(row["metrics"]["concept_recall"], 1, 1)
        self.assert_rate(row["metrics"]["supported_precision"], 1, 1)
        self.assertEqual(len(report["concepts"]["phrase_audit"][0]["phrases"]), 4)

    def test_broad_phrases_never_match_positive_concepts(self):
        frame = self.fixture.frame(concepts={"tree": [(10, 10)], "water": [(10, 11)]})
        self.fixture.predict(frame, ["unsafe area", "non traversable objects", "obstacles"])
        row = self.model_row(self.evaluate())
        self.assert_rate(row["metrics"]["concept_recall"], 0, 2)
        self.assert_rate(row["metrics"]["supported_precision"], 0, 0)

    def test_empty_reference_empty_prediction_rates_are_null(self):
        frame = self.fixture.frame(source="coco")
        self.fixture.predict(frame, [])
        row = self.model_row(self.evaluate(), "coco")
        self.assert_rate(row["metrics"]["concept_recall"], 0, 0)
        self.assert_rate(row["metrics"]["supported_precision"], 0, 0)
        self.assert_rate(row["metrics"]["tiny_concept_recall"], 0, 0)

    def test_tiny_area_threshold_excludes_exact_boundary_but_keeps_one_pixel(self):
        frame = self.fixture.frame(size=(100, 100), concepts={
            "water": [(10, column) for column in range(10)],
            "tree": [(11, column) for column in range(11)],
            "person": [(12, 1)],
        })
        self.fixture.predict(frame, ["water"])
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        row = self.model_row(report)
        self.assert_rate(row["metrics"]["concept_recall"], 1, 3)
        self.assert_rate(row["metrics"]["tiny_concept_recall"], 0, 1)
        self.assertEqual(row["tiny_positive_pairs"], 1)

    def test_saved_error_with_partial_prompts_is_invalid_and_misses_positive(self):
        frame = self.fixture.frame(concepts={"water": [(10, 10)]})
        self.fixture.predict(frame, ["water"], status="error", error_code="generation_truncated")
        report = self.evaluate()
        self.assert_blocked(report)
        self.assert_rate(self.model_row(report)["metrics"]["concept_recall"], 0, 1)
        self.assertEqual(report["validation"]["errors"][0]["raw_prompts"], ["water"])

    def test_raw_exact_duplicates_survive_auditing_after_saved_normalized_deduplication(self):
        frame = self.fixture.frame(concepts={"water": [(10, 10)]})
        prediction = self.fixture.predict(frame, [" PUDDLE ", "WATER"])
        prediction["raw_response"] = '{"prompts":[" PUDDLE ","puddle","WATER"]}'
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        row = self.model_row(report)
        self.assert_rate(row["metrics"]["concept_recall"], 1, 1)
        self.assertEqual(row["normalization_duplicate_count"], 1)
        self.assertEqual(row["mapped_concept_duplicate_count"], 1)
        self.assertEqual(report["concepts"]["phrase_audit"][0]["raw_phrases"], [" PUDDLE ", "puddle", "WATER"])

    def test_invalid_or_inconsistent_raw_response_cannot_claim_success(self):
        invalid = ['{"prompts":["water"],"confidence":0.9}', '{"prompts":["tree"]}',
                   '{"prompts":', '{"prompts":[""]}', '{"prompts":[null]}']
        for index, response in enumerate(invalid):
            with self.subTest(response=response):
                self.fixture = HazardFixture(Path(self.temporary.name) / str(index))
                frame = self.fixture.frame(concepts={"water": [(10, 10)]})
                self.fixture.predict(frame, ["water"])["raw_response"] = response
                report = self.evaluate()
                self.assert_blocked(report)
                self.assert_rate(self.model_row(report)["metrics"]["concept_recall"], 0, 1)

    def test_configured_metadata_pointers_allow_explicit_producer_layouts(self):
        self.fixture.discovery_case()
        identity_fields = ("task_id", "schema_version", "fixture", "policy_hash", "alias_hash",
                           "dataset_fingerprint")
        self.fixture.config["metadata_bindings"] = {}
        for kind, metadata in (("dataset", self.fixture.dataset_metadata),
                               ("model", self.fixture.model_metadata)):
            metadata["identity"] = {field: metadata.pop(field) for field in identity_fields}
            self.fixture.config["metadata_bindings"][kind] = {
                field: f"/identity/{field}" for field in identity_fields}
        self.fixture.model_metadata["inputs"] = {
            "requested": self.fixture.model_metadata.pop("requested_frame_ids")}
        self.fixture.config["metadata_bindings"]["model"]["requested_frame_ids"] = "/inputs/requested"
        report = self.evaluate()
        self.assertEqual(report["validation"]["errors"], [])
        self.assert_rate(self.model_row(report)["metrics"]["concept_recall"], 2, 4)

    def test_pinned_input_signature_detects_same_count_reference_asset_mutation(self):
        self.fixture.discovery_case()
        original = self.evaluate()
        self.fixture.config["expected_input_signature"] = original["identity"]["observed_input_signature"]
        pinned = self.evaluate()
        self.assertEqual(pinned["validation"]["errors"], [])
        self.assertTrue(pinned["identity"]["input_identity_verified"])
        path = self.fixture.references[0]["concept_masks"]["water"]
        self.fixture.mask(path, [(10, 11)])  # One valid pixel remains one; content changed.
        changed = self.evaluate()
        self.assert_blocked(changed)
        self.assertIn("invalid_input_fingerprint", {item["code"] for item in changed["validation"]["errors"]})
        self.assert_rate(self.model_row(changed)["metrics"]["concept_recall"], 2, 4)
        self.assertFalse(changed["identity"]["input_identity_verified"])

    def test_supported_recomputed_dataset_fingerprint_recipe_verifies_actual_inputs(self):
        self.fixture.discovery_case()
        original = self.evaluate()
        observed = original["identity"]["observed_input_signature"]
        self.fixture.config["dataset_fingerprint_recipe"] = "frames_references_assets_v1"
        self.fixture.dataset_metadata["dataset_fingerprint"] = observed
        self.fixture.model_metadata["dataset_fingerprint"] = observed
        verified = self.evaluate()
        self.assertEqual(verified["validation"]["errors"], [])
        self.assertTrue(verified["identity"]["input_identity_verified"])

    def test_malformed_configuration_raises_value_error(self):
        cases = [None, [], {"fixture": "true"}, {"model_keys": []},
                 {"model_keys": ["../../escape"]}, {"model_keys": [MODEL, MODEL]},
                 {"coverage": {"expected_frames": {"rellis": {"test": -1}}}},
                 {"metadata_bindings": {"model": {"task_id": "not_a_pointer"}}},
                 {"segmentation": {"enabled": "yes"}},
                 {"deployment": {"profiles": "not_a_list"}}]
        for malformed in cases:
            with self.subTest(config=malformed):
                with self.assertRaises(ValueError):
                    evaluate(malformed)

    def test_duplicate_json_keys_nonfinite_numbers_and_orphan_ids_are_errors(self):
        self.fixture.discovery_case()
        self.fixture.save()
        path = self.fixture.run / "predictions" / f"{MODEL}.jsonl"
        path.write_text('{"frame_id":"fixture","frame_id":"other"}\n', encoding="utf-8")
        self.assert_blocked(evaluate(self.fixture.config))
        path.write_text('{"frame_id":"fixture","value":NaN}\n', encoding="utf-8")
        self.assert_blocked(evaluate(self.fixture.config))
        orphan = copy.deepcopy(self.fixture.predictions[0])
        orphan["frame_id"] = "rellis:00001:orphan"
        write_jsonl(path, [*self.fixture.predictions, orphan])
        self.assert_blocked(evaluate(self.fixture.config))

    def test_export_round_trip_preserves_rate_triples_nulls_raw_phrase_and_marks_fixture(self):
        self.fixture.discovery_case()
        report = self.evaluate()
        output_dir = self.fixture.run / "evaluation"
        paths = export_report(report, output_dir)
        self.assertIsInstance(paths, dict)
        self.assertTrue(paths)
        for exported in paths.values():
            path = Path(exported)
            self.assertTrue(path.is_file(), path)
            self.assertTrue(path.resolve().is_relative_to(output_dir.resolve()), path)
        document = json.loads((output_dir / "hazard_report.json").read_text("utf-8"))
        self.assertEqual(document, report)
        self.assertTrue(document["fixture"])
        self.assertIn(" PUDDLE ", (output_dir / "hazard_report.json").read_text("utf-8"))
        self.assertTrue(list(output_dir.glob("*.csv")))
        self.assertTrue(list(output_dir.glob("*.jsonl")))

    def test_package_import_does_not_load_models_or_new_sibling_implementations(self):
        script = "\n".join([
            "import sys", f"sys.path.insert(0, {str(COMPONENT / 'src')!r})",
            "import traversability_hazard_evaluation",
            "blocked = ('torch', 'transformers', 'sam3', 'traversability_hazard_data', "
            "'traversability_hazard_inference', 'traversability_hazard_segmentation')",
            "assert not [name for name in sys.modules if name.startswith(blocked)]",
        ])
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_notebook_is_output_cleared_and_thin(self):
        path = COMPONENT / "notebooks/12_hazard_evaluation.ipynb"
        notebook = json.loads(path.read_text("utf-8"))
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(cells)
        for index, cell in enumerate(cells):
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            compile("".join(cell["source"]), f"hazard-notebook-{index}", "exec")
        source = "\n".join("".join(cell["source"]) for cell in cells)
        self.assertIn("traversability_hazard_evaluation", source)
        self.assertNotIn("traversability_hazard_inference", source)
        self.assertNotIn("traversability_hazard_data", source)
        self.assertNotIn("traversability_hazard_segmentation", source)

    def test_notebook_cells_execute_with_fixture_roots_and_export(self):
        self.fixture.discovery_case()
        self.fixture.save()
        notebook = json.loads((COMPONENT / "notebooks/12_hazard_evaluation.ipynb").read_text("utf-8"))
        sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        namespace = {"__name__": "independent_hazard_notebook_fixture"}
        with redirect_stdout(io.StringIO()):
            exec(compile(sources[0], "hazard-notebook-setup", "exec"), namespace)
            namespace["config"].update(self.fixture.config)
            namespace["EXPORT"] = True
            for index, source in enumerate(sources[1:], 1):
                exec(compile(source, f"hazard-notebook-cell-{index}", "exec"), namespace)
        self.assertEqual(namespace["report"]["validation"]["errors"], [])
        self.assertTrue(namespace["report"]["fixture"])
        self.assert_rate(self.model_row(namespace["report"])["metrics"]["concept_recall"], 2, 4)
        self.assertTrue((self.fixture.run / "evaluation/hazard_report.json").is_file())

    def test_default_notebook_executes_with_empty_environment_roots_and_export_off(self):
        run_root = Path(self.temporary.name) / "default-empty-runs"
        data_root = Path(self.temporary.name) / "default-empty-data"
        environment = dict(os.environ)
        environment.update(TRAVERSABILITY_REPO_ROOT=str(COMPONENT),
                           TRAVERSABILITY_RUN_ROOT=str(run_root), TRAVERSABILITY_DATA_ROOT=str(data_root))
        notebook_path = COMPONENT / "notebooks/12_hazard_evaluation.ipynb"
        script = "\n".join([
            "import json, os", f"from pathlib import Path\nnotebook = json.loads(Path({str(notebook_path)!r}).read_text('utf-8'))",
            "namespace = {'__name__': 'default_hazard_notebook_cpu_check'}",
            "for index, cell in enumerate(notebook['cells']):",
            "    if cell['cell_type'] == 'code':",
            "        exec(compile(''.join(cell['source']), f'notebook-cell-{index}', 'exec'), namespace)",
            "assert namespace['EXPORT'] is False",
            "assert not namespace['report']['comparison']['discovery_complete']",
            "assert not Path(os.environ['TRAVERSABILITY_RUN_ROOT']).exists()",
            "assert not Path(os.environ['TRAVERSABILITY_DATA_ROOT']).exists()",
        ])
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                   cwd=COMPONENT, env=environment)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Export disabled", completed.stdout)
        self.assertFalse(run_root.exists())
        self.assertFalse(data_root.exists())

    def test_cli_complete_fixture_and_missing_prediction_exit_codes_and_export(self):
        self.fixture.discovery_case()
        self.fixture.save()
        config_path = Path(self.temporary.name) / "cli-config.json"
        write_json(config_path, self.fixture.config)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(COMPONENT / "src")
        output_dir = self.fixture.run / "evaluation"
        command = [sys.executable, "-m", "traversability_hazard_evaluation", "--config", str(config_path)]
        complete = subprocess.run([*command, "--output-dir", str(output_dir)], capture_output=True,
                                  text=True, cwd=COMPONENT, env=environment)
        self.assertEqual(complete.returncode, 0, complete.stderr)
        self.assertTrue((output_dir / "hazard_report.json").is_file())
        exported = json.loads((output_dir / "hazard_report.json").read_text("utf-8"))
        self.assertTrue(exported["fixture"])
        self.assertFalse(exported["comparison"]["comparison_ready"])
        self.fixture.predictions.pop()
        self.fixture.save()
        incomplete = subprocess.run(command, capture_output=True, text=True, cwd=COMPONENT, env=environment)
        self.assertEqual(incomplete.returncode, 2, incomplete.stderr)
        self.assertIn("missing_requested_predictions", incomplete.stdout)
        self.assertIn(self.fixture.frames[1]["frame_id"], incomplete.stdout)


if __name__ == "__main__":
    unittest.main()
