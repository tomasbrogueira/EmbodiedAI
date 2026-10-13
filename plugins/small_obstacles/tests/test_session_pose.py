import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from plugins.small_obstacles.model_session import ModelSession
from plugins.small_obstacles.scene_pose import PoseHistory
from plugins.small_obstacles.stream_worker import LingBotStream
from plugins.small_obstacles.run_streaming import project_frame


class FakeWorker:
    made = []
    def __init__(self,role,python,config,directory):
        self.role,self.closed,self.pending,self.operations = role,False,False,[]
        self.process = SimpleNamespace(pid=len(self.made)+1,poll=lambda:0 if self.closed else None)
        self.ready={'load_seconds':3.}
        self.made.append(self)
    def call(self,**request):
        self.operations.append(request)
        return {'status':'ok','sequence_reset':True}
    def receive(self):
        self.pending=False
        self.operations.append({'op':'drain'})
    def close(self):
        self.closed=True


class SessionPoseTests(unittest.TestCase):
    def setUp(self):
        FakeWorker.made=[]
        self.directory=tempfile.TemporaryDirectory()
        self.path=Path(self.directory.name)/'config.json'
        self.config={r+'_python':'python' for r in ['lingbot','vlm','sam']}
        self.config['kv_cache_window']=8
        self.path.write_text(json.dumps(self.config))
    def tearDown(self):
        self.directory.cleanup()
    def begin(self,session):
        results=[]
        workers=session.begin(self.path,self.path.parent,lambda role:None,
                              lambda role,worker,result:results.append(result))
        return workers,results
    def test_reuses_workers_and_resets_sequences_without_reloading(self):
        session=ModelSession(FakeWorker)
        first,cold=self.begin(session)
        first[1].pending=True
        session.finish()
        second,warm=self.begin(session)
        self.assertEqual([w.process.pid for w in first],[w.process.pid for w in second])
        self.assertEqual(len(FakeWorker.made),3)
        self.assertTrue(all(r['load_seconds']==0 and r['reused'] for r in warm))
        self.assertTrue(all(not r['reused'] for r in cold))
        self.assertEqual(first[1].operations[0]['op'],'drain')
        self.assertTrue(all(any(o['op']=='reset' for o in w.operations) for w in second))
        session.finish();session.close()
        self.assertFalse(session.loaded)
    def test_active_session_rejects_second_video_and_unload(self):
        session=ModelSession(FakeWorker)
        self.begin(session)
        with self.assertRaisesRegex(RuntimeError,'active video'):self.begin(session)
        with self.assertRaisesRegex(RuntimeError,'Stop'):session.close()
        session.finish();session.close()
    def test_changed_model_settings_and_failed_runs_reload(self):
        session=ModelSession(FakeWorker)
        first,_=self.begin(session);session.finish()
        self.config['kv_cache_window']=4;self.path.write_text(json.dumps(self.config))
        second,cold=self.begin(session)
        self.assertTrue(all(w.closed for w in first))
        self.assertTrue(all(not r['reused'] for r in cold))
        session.finish(healthy=False)
        self.assertTrue(all(w.closed for w in second))
        self.assertFalse(session.loaded)
        third,_=self.begin(session)
        self.assertEqual(len(FakeWorker.made),9)
        session.finish();session.close()
    def test_lingbot_reset_clears_both_history_and_sequence_counters(self):
        cleared=[]
        model=SimpleNamespace(clean_kv_cache=lambda:cleared.append('all caches'),
                              _set_skip_append=lambda value:cleared.append(value))
        runner=object.__new__(LingBotStream)
        runner.model,runner.count,runner.scale_frames=model,100,2
        runner.reset()
        self.assertEqual((runner.count,runner.scale_frames),(0,0))
        self.assertEqual(cleared,['all caches',False])
    def test_camera_marker_agrees_with_official_projection_at_principal_pixel(self):
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_euler('xyz',[.3,.2,-.4]).as_matrix()
        center=np.array([.1,-.2,.3])
        extrinsic=np.c_[rotation.T,-rotation.T@center]
        prediction=dict(depth=np.full((28,28),2,np.float32),depth_conf=np.full((28,28),3,np.float32),
                        images=np.full((28,28,3),100,np.uint8),
                        intrinsic=np.array([[10,0,14],[0,10,14],[0,0,1]],np.float32),extrinsic=extrinsic)
        history=PoseHistory();pose=history.observe(prediction,0,0.)
        points,_,_=project_frame(prediction,np.zeros((28,28),bool))
        np.testing.assert_allclose(points.reshape(28,28,3)[14,14],
            np.asarray(pose['position_world'])+np.asarray(pose['view_direction_world'])*2,atol=1e-6)
        np.testing.assert_allclose(pose['relative_position'],[0,0,0],atol=1e-7)
        prediction['extrinsic']=np.c_[rotation.T,-rotation.T@(center+rotation@np.array([0,0,.2]))]
        second=history.observe(prediction,7,1.)
        np.testing.assert_allclose(second['relative_position'],[0,0,.2],atol=1e-6)
        self.assertEqual(len(second['trajectory_world']),2)
        self.assertEqual(len(PoseHistory().observe(prediction,0,0.)['trajectory_world']),1)

    def test_pose_helper_imports_geometry_in_a_fresh_process(self):
        result = subprocess.run([sys.executable,'-c',
            'import numpy as np; from plugins.small_obstacles.scene_pose import PoseHistory; '
            'p=dict(extrinsic=np.c_[np.eye(3),[0,0,0]],intrinsic=np.eye(3),images=np.zeros((28,28,3))); '
            'assert PoseHistory().observe(p,0,0.)["position_world"]==[0.,0.,0.]'],
            cwd=Path(__file__).resolve().parents[3],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':
    unittest.main()
