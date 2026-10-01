"""Pre-inference pins and unchanged pre-integration artifact preservation."""

from pathlib import Path
import tempfile
import unittest

from common import config, file_snapshot, read_json, read_jsonl, write_json
from traversability_hazard_data import prepare_run, validate_run
from traversability_hazard_data.storage import fingerprint, input_signature


class IntegratedInputPins(unittest.TestCase):
    def test_new_signature_is_prepared_before_predictions_and_cannot_be_rebound(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            run = prepare_run(cfg, fixture=True)
            metadata = read_json(run / "metadata/dataset.json")
            frames, refs = read_jsonl(run / "frames.jsonl"), read_jsonl(run / "references.jsonl")
            self.assertFalse((run / "predictions").exists())
            self.assertEqual(metadata["selected_frame_ids"], [row["frame_id"] for row in frames])
            self.assertEqual(metadata["observed_input_signature"], input_signature(frames, refs, metadata["assets"]))
            metadata["observed_input_signature"] = "0" * 64
            metadata["dataset_fingerprint"] = fingerprint(metadata, frames, refs)
            write_json(run / "metadata/dataset.json", metadata)
            with self.assertRaisesRegex(ValueError, "input signature"):
                validate_run(run, cfg["data_root"])

    def test_unchanged_pre_integration_run_is_not_rewritten_with_new_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary)
            run = prepare_run(cfg, fixture=True)
            metadata = read_json(run / "metadata/dataset.json")
            del metadata["observed_input_signature"], metadata["selected_frame_ids"]
            metadata["dataset_fingerprint"] = fingerprint(
                metadata, read_jsonl(run / "frames.jsonl"), read_jsonl(run / "references.jsonl"))
            write_json(run / "metadata/dataset.json", metadata)
            before = file_snapshot(run)
            prepare_run(cfg, fixture=True)
            self.assertEqual(before, file_snapshot(run))
            self.assertTrue(validate_run(run, cfg["data_root"])["fixture"])

    def test_protocol_fixture_cannot_create_an_unmarked_real_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = dict(config(temporary), fixture_profile="protocol")
            with self.assertRaisesRegex(ValueError, "fixture=True"):
                prepare_run(cfg)
            self.assertFalse(Path(cfg["run_root"]).exists())


if __name__ == "__main__":
    unittest.main()
