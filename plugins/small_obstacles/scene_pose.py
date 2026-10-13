"""Camera poses in exactly the coordinate convention used by map projection."""
import numpy as np

FOLLOW_DISTANCE = 1.2
FOLLOW_HEIGHT = .6
FOLLOW_FOV = 80.


def camera_view(pose, extent, distance=FOLLOW_DISTANCE, height=FOLLOW_HEIGHT, fov_degrees=FOLLOW_FOV):
    """Third-person view behind/above the camera in its local map-coordinate frame."""
    from scipy.spatial.transform import Rotation
    quaternion = np.asarray(pose['quaternion_wxyz'])
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    camera_position = np.asarray(pose['position_world'], dtype=float)
    scale = max(float(extent), 1e-3)
    forward, up = rotation[:, 2], -rotation[:, 1]
    return {'position': camera_position-forward*scale*distance+up*scale*height,
            'look_at': camera_position+forward*scale*.3,
            'up': up, 'fov': np.deg2rad(fov_degrees), 'near': max(scale*.001, 1e-5)}


class CameraFollower:
    def __init__(self, server):
        self.server = server
        self.reset()

    def reset(self):
        self.initial_key = None
        self.client_keys = {}
        self.following = None

    def update(self, pose, extent, follow=True, distance=FOLLOW_DISTANCE, height=FOLLOW_HEIGHT, fov_degrees=FOLLOW_FOV):
        if not pose:
            return
        if follow and self.following is not True:
            self.client_keys = {}
        self.following = follow
        key = (pose['source_index'], pose['source_time'], float(extent), distance, height, fov_degrees)
        if self.initial_key is not None and not follow:
            return
        view = camera_view(pose, extent, distance, height, fov_degrees)
        if key != self.initial_key:
            initial = self.server.initial_camera
            initial.position, initial.look_at = view['position'], view['look_at']
            initial.up, initial.fov, initial.near = view['up'], view['fov'], view['near']
            self.initial_key = key
        clients = self.server.get_clients()
        self.client_keys = {client_id: old for client_id, old in self.client_keys.items() if client_id in clients}
        for client_id, client in clients.items():
            if not client.camera.update_timestamp or self.client_keys.get(client_id) == key:
                continue
            # Viser position updates shift look_at: assign the desired target next.
            with client.atomic():
                client.camera.position = view['position']
                client.camera.look_at = view['look_at']
                client.camera.up_direction = view['up']
                client.camera.fov = view['fov']
                client.camera.near = view['near']
            self.client_keys[client_id] = key


class PoseHistory:
    def __init__(self):
        self.initial = None
        self.positions = []

    def observe(self, prediction, source_index, source_time):
        from pathlib import Path
        import sys
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / 'orchestrator'))
        sys.path.insert(0, str(root / 'lingbot-map'))
        from io_utils import load_extrinsic, load_intrinsic, load_rgb
        from lingbot_map.utils.geometry import closed_form_inverse_se3
        from scipy.spatial.transform import Rotation
        # The official depth helper inverts the supplied saved extrinsic.
        # Perform the SAME operation to locate its camera; mapping is unchanged.
        world_from_camera = closed_form_inverse_se3(load_extrinsic(prediction['extrinsic'])[None])[0]
        if self.initial is None:
            self.initial = world_from_camera.copy()
        rotation = world_from_camera[:3, :3]
        position = world_from_camera[:3, 3]
        relative_rotation = self.initial[:3, :3].T @ rotation
        relative_position = self.initial[:3, :3].T @ (position-self.initial[:3, 3])
        quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        forward = rotation[:, 2]/np.linalg.norm(rotation[:, 2])
        self.positions.append(position.tolist())
        intrinsic = load_intrinsic(prediction['intrinsic'])
        h,w = load_rgb(prediction['images']).shape[:2]
        return {'source_index': source_index, 'source_time': source_time,
            'position_world': position.tolist(), 'quaternion_wxyz': quaternion.tolist(),
            'view_direction_world': forward.tolist(), 'relative_position': relative_position.tolist(),
            'relative_rotation_degrees': Rotation.from_matrix(relative_rotation).as_rotvec(degrees=True).tolist(),
            'vertical_fov': float(2*np.arctan(h/(2*intrinsic[1,1]))), 'aspect': w/h,
            'trajectory_world': list(self.positions),
            'convention': 'Camera pose in map coordinates; OpenCV +Z forward, +X right, +Y down. Scale is model-relative.'}


class PoseOverlay:
    def __init__(self, server):
        self.server, self.handles, self.key = server, [], None

    def clear(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles, self.key = [], None

    def show_camera(self, visible):
        # Camera marker visibility can change independently of the trajectory.
        for handle in self.handles[:4]:
            handle.visible = visible

    def update(self, pose, extent):
        if not pose:
            return
        key = (pose['source_index'], extent)
        if key == self.key:
            return
        scale = max(extent*.06, 1e-4)
        scene = self.server.scene
        position = np.asarray(pose['position_world'])
        orientation = tuple(pose['quaternion_wxyz'])
        # Re-adding these names replaces the existing scene objects.
        self.handles = [
            scene.add_frame('/current_camera', position=position, wxyz=orientation,
                            axes_length=scale, axes_radius=scale*.04),
            scene.add_camera_frustum('/current_camera/frustum', fov=pose['vertical_fov'],
                aspect=pose['aspect'], scale=scale, color=(255,180,0), thickness=1.5, thickness_units='screen'),
            scene.add_line_segments('/view_direction',
                points=np.array([[position,position+np.asarray(pose['view_direction_world'])*scale*3]],np.float32),
                colors=(255,180,0),thickness=3,thickness_units='screen'),
            scene.add_label('/camera_label', f'Current camera · {pose["source_time"]:.2f}s', position=position)]
        positions = np.asarray(pose['trajectory_world'],np.float32)
        if len(positions)>1:
            self.handles.append(scene.add_line_segments('/camera_path',
                points=np.stack([positions[:-1],positions[1:]],axis=1), colors=(0,180,255),thickness=2,thickness_units='screen'))
        self.key = key
