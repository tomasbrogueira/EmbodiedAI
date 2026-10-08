"""Only the conditional semantic-center display-sampling caption delta."""
from copy import deepcopy
import textwrap
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import export_native_recording as package


class SamplingCaptionTests(unittest.TestCase):
    def fixture(self,union=6001,selected=6000):
        return {
            "native_config":{
                "optional_added_overlays":{"semantic_positive_evidence":True},
                "semantic_overlay_sampling":{"union_cells":union,"selected_cells":selected,
                                             "cap":6000,"saved_evidence_changed":False}},
            "native":{
                "coverage":[{"timestamp_ns":0,"timestamp_provenance":{},"queries":[]}],
                "geometry_identity":{"units":"reconstruction"},
                "original_planning":{"availability":"blocked"},
                "research_plan":{"status":"no_path"},
                "capture":{"frames":[{"research_route_visible":False}]}},
            "segmentation":{},"segmentation_config":{},
            "native_pixels":[np.full((800,800,3),57,np.uint8)],
            "segmentation_pixels":[np.full((800,800,3),114,np.uint8)]}

    def test_sampling_notice_reaches_native_and_combined_pixels_and_player_facts(self):
        data = self.fixture()
        before = deepcopy(data)
        calls = []
        original = package._draw
        def draw(image,xy,text,**kwargs):
            calls.append((xy,text,kwargs))
            original(image,xy,text,**kwargs)
        with patch.object(package,"_draw",side_effect=draw):
            panels = package.panels(data,0)
        footer = [text for xy,text,_ in calls if xy==(18,914)]
        self.assertIn("Semantic-center overlay is display-sampled",footer[0])
        self.assertIn("saved masks/evidence unchanged",footer[0])
        self.assertNotIn("display-sampled",footer[1])
        self.assertTrue(any("display-sampled" in fact and "saved masks and evidence are unchanged" in fact
                            for fact in package.facts(data)))
        np.testing.assert_array_equal(np.asarray(panels["native_map"])[104:904],data["native_pixels"][0])
        np.testing.assert_array_equal(np.asarray(panels["combined"])[104:904,800:],data["native_pixels"][0])
        np.testing.assert_array_equal(data["native_pixels"][0],before["native_pixels"][0])
        self.assertEqual(data["native_config"],before["native_config"])
        wrapped = "\n".join(textwrap.wrap(footer[0],width=88))
        bounds = ImageDraw.Draw(Image.new("RGB",(800,1000))).multiline_textbbox(
            (18,914),wrapped,font=package._font(16),spacing=3)
        self.assertLessEqual(bounds[2],800)
        self.assertLessEqual(bounds[3],1000)
        self.assertTrue(footer[0].isascii())

    def test_unsampled_missing_disabled_and_invalid_counts_have_no_notice(self):
        cases = [self.fixture(6000,6000),self.fixture(0,0),self.fixture(None,6000),
                 self.fixture(True,0),self.fixture("6001",6000),self.fixture(6001,-1)]
        missing = self.fixture()
        missing["native_config"].pop("semantic_overlay_sampling")
        cases.append(missing)
        disabled = self.fixture()
        disabled["native_config"]["optional_added_overlays"]["semantic_positive_evidence"] = False
        cases.append(disabled)
        for data in cases:
            with self.subTest(config=data["native_config"]):
                self.assertFalse(package.semantic_overlay_is_display_sampled(data))
                with patch.object(package,"_draw") as draw:
                    package.panels(data,0)
                footers = [call.args[2] for call in draw.call_args_list if call.args[1]==(18,914)]
                self.assertTrue(all("display-sampled" not in text for text in footers))
                self.assertFalse(any("display-sampled" in fact for fact in package.facts(data)))


if __name__=="__main__":
    unittest.main()
