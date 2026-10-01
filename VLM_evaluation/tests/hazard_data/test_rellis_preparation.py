"""Exercise real preparation logic using small synthetic original-ID trees."""

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from common import config, file_snapshot, read_json, read_jsonl
from traversability_hazard_data import prepare_run, validate_run


def write_sequence(source, sequence="00000", count=20, *, labels_count=None):
    """A publisher-shaped tree, explicitly synthetic and never experiment evidence."""
    source = Path(source)
    rgb = source / sequence / "pylon_camera_node"
    labels = source / sequence / "pylon_camera_node_label_id"
    rgb.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    labels_count = count if labels_count is None else labels_count
    for number in range(1, count + 1):
        # Nonzero varying pixels make every original RGB byte hash distinct.
        values = np.zeros((10, 10, 3), dtype=np.uint8)
        values[:, :, 0] = number
        values[:, :, 1] = int(sequence) + 70
        values[:, :, 2] = np.arange(10, dtype=np.uint8)
        Image.fromarray(values).save(rgb / f"frame_{number}.png")
        if number <= labels_count:
            ids = np.full((10, 10), 7, dtype=np.uint8)
            ids.flat[:(4 + (number - 1) % 3)] = 0
            ids.flat[-1] = 17
            Image.fromarray(ids).save(labels / f"frame_{number}.png")
    return rgb, labels


class RellisPreparationTests(unittest.TestCase):
    def test_natural_uniform_selection_and_rellis_split_are_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "synthetic-source"
            for sequence in ("00000", "00001"):
                write_sequence(source, sequence, 31)
            cfg = config(base)
            cfg["rellis"] = {"source_dir": str(source)}
            run = prepare_run(cfg)
            frames = read_jsonl(run / "frames.jsonl")
            selection = read_json(run / "metadata/dataset.json")["selection"]["rellis"]
            expected = [f"frame_{1 + math.floor(index * 30 / 19 + .5)}" for index in range(20)]
            for sequence, split in (("00000", "development"), ("00001", "test")):
                selected = [row for row in frames if row["sequence_id"] == sequence]
                self.assertEqual({row["frame_id"].split(":")[-1] for row in selected}, set(expected))
                self.assertEqual([row["stem"] for row in selection[sequence]["selected"]], expected)
                self.assertEqual({row["split"] for row in selected}, {split})
                self.assertTrue(all(row["timestamp_s"] is None for row in selected))
                self.assertTrue(all((row["width"], row["height"]) == (10, 10) for row in selected))
            self.assertEqual(len(frames), 40)
            self.assertEqual(len({row["frame_id"] for row in frames}), 40)
            self.assertTrue(validate_run(run, cfg["data_root"])["valid"])
            first = file_snapshot(run)
            prepare_run(cfg)
            self.assertEqual(file_snapshot(run), first)

    def test_coverage_threshold_applies_only_to_absence_and_retains_one_pixel_positives(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "synthetic-source"
            write_sequence(source)
            cfg = config(base)
            cfg["rellis"] = {"source_dir": str(source)}
            run = prepare_run(cfg)
            references = read_jsonl(run / "references.jsonl")
            self.assertEqual(len(references), 20)
            for row in references:
                number = int(row["frame_id"].rsplit("_", 1)[-1])
                ignored = 4 + (number - 1) % 3
                self.assertEqual(row["absence_scoring_eligible"], ignored <= 5)
                self.assertEqual(row["present_concepts"], ["person"])
                self.assertEqual(row["concept_pixel_counts"], {"person": 1})

    def test_rgb_only_and_underfilled_sequences_remain_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "synthetic-source"
            write_sequence(source, "00000", 20, labels_count=0)
            write_sequence(source, "00001", 20, labels_count=19)
            cfg = config(base)
            cfg["rellis"] = {"source_dir": str(source)}
            run = prepare_run(cfg)
            self.assertEqual(read_jsonl(run / "frames.jsonl"), [])
            text = str(read_json(run / "metadata/dataset.json")["coverage"])
            self.assertIn("00000", text)
            self.assertIn("00001", text)
            self.assertIn("19", text)
            self.assertTrue(validate_run(run, cfg["data_root"])["valid"])

    def test_missing_sources_can_fill_before_downstream_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "synthetic-source"
            source.mkdir()
            cfg = config(base)
            cfg["rellis"] = {"source_dir": str(source)}
            run = prepare_run(cfg)
            self.assertEqual(read_jsonl(run / "frames.jsonl"), [])
            write_sequence(source)
            prepare_run(cfg)
            self.assertEqual(len(read_jsonl(run / "frames.jsonl")), 20)
            self.assertTrue(validate_run(run, cfg["data_root"])["valid"])

    def test_downstream_files_freeze_additions_to_pending_dataset(self):
        for downstream in ("predictions/model.jsonl", "evaluation/hazard_report.json",
                           "segmentation/selection.json", "benchmark/profile/samples.jsonl",
                           "metadata/model.json"):
            with self.subTest(downstream=downstream), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                source = base / "synthetic-source"
                source.mkdir()
                cfg = config(base)
                cfg["rellis"] = {"source_dir": str(source)}
                run = prepare_run(cfg)
                artifact = run / downstream
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text("{}\n", encoding="utf-8")
                before = file_snapshot(run)
                write_sequence(source)
                with self.assertRaises((ValueError, OSError)):
                    prepare_run(cfg)
                self.assertEqual(file_snapshot(run), before)
                self.assertEqual(read_jsonl(run / "frames.jsonl"), [])

    def test_new_source_frames_do_not_replace_a_frozen_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "synthetic-source"
            rgb, labels = write_sequence(source, count=31)
            cfg = config(base)
            cfg["rellis"] = {"source_dir": str(source)}
            run = prepare_run(cfg)
            before = file_snapshot(run)
            Image.fromarray(np.full((10, 10, 3), 222, dtype=np.uint8)).save(rgb / "frame_0.png")
            Image.fromarray(np.full((10, 10), 7, dtype=np.uint8)).save(labels / "frame_0.png")
            prepare_run(cfg)
            self.assertEqual(file_snapshot(run), before)

    def test_changed_completed_source_bytes_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "synthetic-source"
            rgb, _ = write_sequence(source)
            cfg = config(base)
            cfg["rellis"] = {"source_dir": str(source)}
            run = prepare_run(cfg)
            before = file_snapshot(run)
            Image.fromarray(np.full((10, 10, 3), 255, dtype=np.uint8)).save(rgb / "frame_1.png")
            with self.assertRaises((ValueError, OSError)):
                prepare_run(cfg)
            self.assertEqual(file_snapshot(run), before)

    def test_misaligned_selected_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "synthetic-source"
            _, labels = write_sequence(source)
            Image.fromarray(np.full((9, 10), 7, dtype=np.uint8)).save(labels / "frame_1.png")
            cfg = config(base)
            cfg["rellis"] = {"source_dir": str(source)}
            with self.assertRaises((ValueError, OSError)):
                prepare_run(cfg)


if __name__ == "__main__":
    unittest.main()
