import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from plugins.small_obstacles.run_existing_pipeline import adapt, selected_frames


class ExistingMapperAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.predictions = self.root/'predictions'
        self.masks = self.root/'masks'
        self.predictions.mkdir()
        self.masks.mkdir()
        self.rgb = np.full((28,28,3),30,dtype=np.uint8)
        self.mask = np.zeros((28,28),dtype=bool)
        self.mask[3,4] = True
        self.names = ['frame_000002.npz','frame_000004.npz']
        for name in self.names:
            np.savez(self.predictions/name,images=self.rgb,depth=np.ones((28,28)),
                     depth_conf=np.full((28,28),3), intrinsic=np.eye(3),
                     extrinsic=np.c_[np.eye(3),np.zeros(3)])
            np.savez(self.masks/name,out_binary_masks=self.mask,
                     rgb_sha256=np.asarray(hashlib.sha256(self.rgb.tobytes()).hexdigest()))
        (self.masks/'manifest.json').write_text(json.dumps({'schema_version':1,'frames':self.names}))
        self.selection=self.root/'selection.json'
        self.selection.write_text(json.dumps({'frames':self.names}))
        self.args=SimpleNamespace(lingbot_dir=self.predictions,masks_dir=self.masks,
                                 output_dir=self.root/'mapper_masks',video_name='test',selection_file=self.selection)

    def test_selection_uses_indices_not_list_positions(self):
        (self.predictions/'frame_000005.npz').touch()
        self.assertEqual([p.name for p in selected_frames(self.predictions)],self.names)

    def test_export_and_original_mapper(self):
        adapt(self.args)
        from semantic_map import build_semantic_point_cloud
        config=SimpleNamespace(frame_step=2,pixel_step=1,lingbot_dir=self.predictions,
            sam_dir=self.args.output_dir,video_name='test',prompt='obstacles',video_path=self.root/'test.mp4',
            min_depth_conf=2.5,max_depth=25,segment_color=np.array([255,0,0],dtype=np.uint8),
            patch_size=14,out_file=self.root/'map.ply')
        build_semantic_point_cloud(config)
        raw=config.out_file.read_bytes()
        header, payload=raw.split(b'end_header\n')
        self.assertIn(b'element vertex 1568',header)
        dtype=np.dtype([(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ('r','g','b')])
        points=np.frombuffer(payload,dtype=dtype)
        self.assertEqual(int((points['r']==255).sum()),2)
        red=points[points['r']==255]
        np.testing.assert_allclose(red['x'],4)
        np.testing.assert_allclose(red['y'],3)
        np.testing.assert_allclose(red['z'],1)

    def test_changed_rgb_rejected(self):
        np.savez(self.predictions/self.names[0],images=self.rgb+1,
                 depth=np.ones((28,28)),depth_conf=np.ones((28,28)))
        with self.assertRaisesRegex(ValueError,'RGB mismatch'):
            adapt(self.args)
