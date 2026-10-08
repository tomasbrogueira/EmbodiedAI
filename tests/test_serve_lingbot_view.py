"""Display-only fitting and Linux ownership proofs for native CPU previews."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

import serve_lingbot_view as native


class DisplayTests(unittest.TestCase):
    def test_camera_fits_points_in_original_coordinates_without_mutation(self):
        grid = np.array([[x,y,z] for x in (10.,20.) for y in (-4.,4.) for z in (30.,40.)])
        saved = grid.copy()
        viewer = types.SimpleNamespace(all_steps=[0,1],vis_pts_list=[grid,grid.copy()],
                                       cam_dict={"R":{0:np.eye(3)},"t":{0:np.zeros(3)}})
        pose = native.native_display_pose(viewer)
        np.testing.assert_array_equal(grid,saved)
        np.testing.assert_allclose(pose["pivot"],[15,0,35])
        np.testing.assert_allclose(pose["up_direction"],[0,-1,0])
        self.assertFalse(pose["geometry_changed"])
        self.assertGreater(pose["distance_native_units"],np.linalg.norm([10,8,10])/2)
        self.assertLessEqual(pose["sample_count"],pose["sample_cap"])

    def test_fit_uses_bounded_samples_and_rejects_empty_native_cloud(self):
        points = np.arange(90000,dtype=float).reshape(-1,3)
        viewer = types.SimpleNamespace(all_steps=[0],vis_pts_list=[points],cam_dict={"R":{0:np.eye(3)}})
        self.assertLessEqual(native.native_display_pose(viewer,sample_cap=30)["sample_count"],30)
        viewer.vis_pts_list = [np.full((3,3),np.nan)]
        with self.assertRaisesRegex(RuntimeError,"no finite"):
            native.native_display_pose(viewer)

    def test_constructor_hides_axes_by_suppressing_native_camera_creation(self):
        server = Mock()
        server.get_host.return_value = "127.0.0.1"
        server.get_port.return_value = 8891
        factory = Mock(return_value=server)
        module = types.SimpleNamespace(viser=types.SimpleNamespace(ViserServer=factory))
        def construct(**kwargs):
            self.assertFalse(kwargs["show_camera"])
            module.viser.ViserServer(port=8891,host="0.0.0.0")
            return types.SimpleNamespace(server=server,downsample_slider=types.SimpleNamespace(),
                                         **{name:types.SimpleNamespace() for name in
                                            ("screenshot_button","glb_export_button","save_video_button")})
        module.PointCloudViewer = construct
        native.construct_loopback_viewer(module,{},8891,10,1.5,.001,hide_native_cameras=True)
        factory.assert_called_once_with(port=8891,host="127.0.0.1")
        self.assertIs(module.viser.ViserServer,factory)

    def test_alternate_port_is_stopped_and_rejected(self):
        server = Mock()
        server.get_host.return_value = "127.0.0.1"
        server.get_port.return_value = 8892
        factory = Mock(return_value=server)
        module = types.SimpleNamespace(viser=types.SimpleNamespace(ViserServer=factory))
        def construct(**kwargs):
            return types.SimpleNamespace(server=module.viser.ViserServer(port=8891))
        module.PointCloudViewer = construct
        with self.assertRaisesRegex(RuntimeError,"exact requested"):
            native.construct_loopback_viewer(module,{},8891,10,1.5,.001)
        server.stop.assert_called_once()
        self.assertIs(module.viser.ViserServer,factory)

    def test_stopped_encoder_does_not_spawn(self):
        stop = threading.Event()
        stop.set()
        with tempfile.TemporaryDirectory() as temporary, patch.object(native,"os",types.SimpleNamespace(name="posix")), \
                patch.object(native.subprocess,"Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError,"before launch"):
                native.OwnedEncoder(stop).run(["unused"],Path(temporary)/"encoder.log",time.monotonic()+1)
            spawn.assert_not_called()


@unittest.skipUnless(sys.platform.startswith("linux"),"actual process-group/flock proof requires Linux")
class LinuxOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def harness(self,mode):
        marker = self.directory/(mode+".pid")
        child_code = ("import os,signal,time; from pathlib import Path; "
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                      f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)")
        code = f'''import json,os,signal,sys,threading,time
from pathlib import Path
from serve_lingbot_view import OwnedEncoder
stop=threading.Event()
signal.signal(signal.SIGTERM,lambda *_:stop.set())
signal.signal(signal.SIGHUP,lambda *_:stop.set())
encoder=OwnedEncoder(stop)
errors=[]
def run():
    try:
        encoder.run([sys.executable,"-c",{child_code!r}],Path({str(self.directory/(mode+".log"))!r}),time.monotonic()+({.3 if mode=="deadline" else 30}))
    except BaseException as error:
        errors.append(str(error))
worker=threading.Thread(target=run,daemon=True)
worker.start()
if {mode!r}=="watchdog":
    until=time.monotonic()+5
    while not Path({str(marker)!r}).exists() and time.monotonic()<until: time.sleep(.01)
    encoder.cancel()
    os._exit(124)
worker.join(8)
if worker.is_alive(): raise RuntimeError("encoder worker failed to stop")
print(json.dumps({{"cleanup":encoder.cleanup,"errors":errors}}),flush=True)
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(native.__file__).parent)
        process = subprocess.Popen([sys.executable,"-c",code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                   text=True,env=environment)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        until = time.monotonic()+5
        while not marker.exists() and process.poll() is None and time.monotonic()<until:
            time.sleep(.01)
        self.assertTrue(marker.exists(),"owned encoder did not initialize")
        child = int(marker.read_text())
        self.assertEqual(os.getpgid(child),child,"encoder is not its own isolated group")
        if mode=="signal":
            process.send_signal(signal.SIGTERM)
        elif mode=="hangup":
            process.send_signal(signal.SIGHUP)
        stdout,stderr = process.communicate(timeout=8)
        self.assertEqual(process.returncode,124 if mode=="watchdog" else 0,stderr)
        with self.assertRaises(ProcessLookupError):
            os.killpg(child,0)
        if mode!="watchdog":
            proof = json.loads(stdout)
            self.assertTrue(proof["cleanup"]["reaped"])
            self.assertTrue(proof["cleanup"]["group_gone"])
            self.assertEqual(proof["cleanup"]["returncode"],-signal.SIGKILL)
            self.assertTrue(proof["errors"])

    def test_sigterm_cancels_and_reaps_running_encoder(self):
        self.harness("signal")

    def test_sighup_cancels_and_reaps_running_encoder(self):
        self.harness("hangup")

    def test_deadline_cancels_and_reaps_running_encoder(self):
        self.harness("deadline")

    def test_watchdog_cancels_encoder_before_emergency_exit(self):
        self.harness("watchdog")

    def test_worker_exception_cancels_and_reaps_owned_encoder(self):
        marker = self.directory/"exception.pid"
        class FailingStop:
            def is_set(self):
                return False
            def wait(self,_):
                until = time.monotonic()+3
                while not marker.exists() and time.monotonic()<until:
                    time.sleep(.01)
                raise RuntimeError("forced worker failure after encoder launch")
        encoder = native.OwnedEncoder(FailingStop())
        child = ("import os,signal,time; from pathlib import Path; "
                 "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                 f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)")
        with self.assertRaisesRegex(RuntimeError,"forced worker failure"):
            encoder.run([sys.executable,"-c",child],self.directory/"exception.log",time.monotonic()+10)
        self.assertTrue(encoder.cleanup["reaped"])
        self.assertTrue(encoder.cleanup["group_gone"])
        with self.assertRaises(ProcessLookupError):
            os.killpg(int(marker.read_text()),0)

    def test_cancellation_signals_whole_owned_group_and_reaps_leader(self):
        marker = self.directory/"descendants.json"
        child = f'''import json,os,signal,subprocess,sys,time
from pathlib import Path
descendant=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])
def stop(*_):
    descendant.wait(timeout=2)
    sys.exit(0)
signal.signal(signal.SIGTERM,stop)
Path({str(marker)!r}).write_text(json.dumps([os.getpid(),descendant.pid]))
time.sleep(60)
'''
        stop = threading.Event()
        encoder = native.OwnedEncoder(stop)
        failures = []
        def run():
            try:
                encoder.run([sys.executable,"-c",child],self.directory/"descendants.log",time.monotonic()+10)
            except RuntimeError as error:
                failures.append(str(error))
        worker = threading.Thread(target=run)
        worker.start()
        until = time.monotonic()+3
        while not marker.exists() and time.monotonic()<until:
            time.sleep(.01)
        self.assertTrue(marker.exists())
        leader,descendant = json.loads(marker.read_text())
        self.assertEqual(os.getpgid(descendant),leader)
        stop.set()
        worker.join(4)
        self.assertFalse(worker.is_alive())
        self.assertTrue(failures)
        self.assertTrue(encoder.cleanup["reaped"])
        self.assertTrue(encoder.cleanup["group_gone"])
        for pid in (leader,descendant):
            with self.assertRaises(ProcessLookupError):
                os.kill(pid,0)

    def test_lease_rejects_collision_and_releases_after_close(self):
        source = self.directory/"lingbot-map"
        source.mkdir()
        first,second = native.NativeLease(source,8891),native.NativeLease(source,8891)
        first.acquire()
        self.addCleanup(first.close)
        with self.assertRaisesRegex(RuntimeError,"Another native"):
            second.acquire()
        first.close()
        second.acquire()
        second.close()


if __name__=="__main__":
    unittest.main()
