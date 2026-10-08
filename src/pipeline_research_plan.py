"""Offline research illustration; assumptions never become calibrated geometry.

This helper reads an immutable completed run and an existing numeric geometry
archive. It creates a separate posthoc plan, not a pipeline planning result.
The common costmap's numerical body is reused in an isolated function namespace
with a research-only input validator; the common module and its verified-input
gate are never patched. No model, GPU, network or producer is invoked.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import FunctionType

import numpy as np

from path_mapping.runner import GEOMETRY_KEYS, geometry_fingerprint, validate_geometry
from pipeline_common import planning as common


DEFAULT_ASSUMPTIONS = {
    "schema_version": 1,
    "research_illustration": True,
    "assumption_id": "research_illustration_v1",
    "metres_per_native_unit": 1.0,
    "robot": {
        "version": "assumed_small_robot_v1", "fixture": True,
        "footprint_radius": .10, "height": .35, "clearance": .02,
        "max_slope_degrees": 25., "max_step": .08, "max_roughness": .05,
        "min_support_points": 3, "unknown_rule": "blocked", "semantic_costs": {},
    },
    "grid_resolution_m": .10,
    "max_grid_cells": 250_000,
    "ground_estimation": {
        "mode": "camera_up_constrained_ransac", "max_up_deviation_degrees": 35.,
        "plane_tolerance_m": .06, "iterations": 160, "max_fit_points": 6000,
        "min_inliers": 30, "min_inlier_fraction": .03,
        "max_camera_below_plane_m": .05, "random_seed": 0,
    },
    "endpoint_policy": "camera_projected_then_largest_observed_component",
    "minimum_endpoint_separation_m": .25,
}
LABEL = "Geometry-only research illustration — assumed scale, ground direction and robot; no safety claim"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _positive(record, key, zero=False):
    return common._positive_number(record, key, zero=zero)


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(name + " must be a finite XYZ vector")
    return array


def validate_assumptions(value):
    value = deepcopy(value)
    if value.get("research_illustration") is not True or value.get("schema_version") != 1:
        raise ValueError("Explicit schema-1 research_illustration=true assumptions required")
    if not value.get("assumption_id"):
        raise ValueError("Assumptions need a distinct assumption_id")
    if "up_vector_native" in value and np.linalg.norm(_vector(value["up_vector_native"], "up_vector_native")) < 1e-12:
        raise ValueError("up_vector_native must be nonzero")
    if "require_voxel_terrain" in value and type(value["require_voxel_terrain"]) is not bool:
        raise ValueError("require_voxel_terrain must be boolean")
    for key, declarations in (("level_reference", {"user_declared_level_floor"}),
                              ("camera_height_reference", {"user_assumed_camera_height", "measured_camera_height"})):
        if key not in value:
            continue
        reference = value[key]
        if (not isinstance(reference, dict) or reference.get("declaration") not in declarations
                or not isinstance(reference.get("recording_id"), str) or not reference["recording_id"]
                or not isinstance(reference.get("geometry_fingerprint"), str)
                or len(reference["geometry_fingerprint"]) != 64
                or any(c not in "0123456789abcdef" for c in reference["geometry_fingerprint"])
                or not isinstance(reference.get("evidence"), str) or not reference["evidence"].strip()):
            raise ValueError(key + " requires an explicit declaration, recording/geometry binding and evidence")
        if key == "level_reference":
            if "up_vector_native" in value:
                raise ValueError("Choose level_reference or up_vector_native; do not silently override either")
            if ("normal_native" in reference) != ("point_native" in reference):
                raise ValueError("A level reference needs both normal_native and point_native")
            if "normal_native" in reference:
                if np.linalg.norm(_vector(reference["normal_native"], "level_reference normal_native")) < 1e-12:
                    raise ValueError("Level reference normal must be nonzero")
                _vector(reference["point_native"], "level_reference point_native")
        else:
            _positive(reference, "height_metres")
    for key in ("metres_per_native_unit", "grid_resolution_m", "minimum_endpoint_separation_m"):
        _positive(value, key)
    if type(value.get("max_grid_cells")) is not int or not 1 <= value["max_grid_cells"] <= 4_000_000:
        raise ValueError("max_grid_cells must be an integer in [1,4000000]")
    robot = value.get("robot")
    if not isinstance(robot, dict) or robot.get("fixture") is not True or not robot.get("version"):
        raise ValueError("Research robot must have explicit synthetic fixture=true and version")
    if robot.get("unknown_rule") != "blocked" or robot.get("semantic_costs") != {}:
        raise ValueError("Research geometry requires unknown_rule=blocked and no semantic repair/reward")
    for key in ("footprint_radius", "height", "clearance", "max_slope_degrees", "max_step", "max_roughness"):
        _positive(robot, key, zero=key != "height")
    if robot["max_slope_degrees"] >= 90:
        raise ValueError("Maximum slope must be below 90 degrees")
    if type(robot.get("min_support_points")) is not int or robot["min_support_points"] < 3:
        raise ValueError("min_support_points must be an integer >=3")
    if value.get("endpoint_policy") not in {
        "camera_projected", "camera_projected_then_largest_observed_component",
        "largest_observed_component", "recorded_forward_corridor",
    }:
        raise ValueError("Unknown research endpoint policy")
    if value['endpoint_policy'] == 'recorded_forward_corridor':
        mission=value.get('forward_mission', {})
        for key in ('camera_lookahead_m','max_endpoint_adjustment_m','minimum_progress_m'):
            _positive(mission,key)
    ground = value.get("ground_estimation")
    if not isinstance(ground, dict):
        raise ValueError("Explicit ground_estimation policy required")
    if ground.get("mode") == "explicit_plane":
        common._basis(_vector(ground.get("normal_native"), "normal_native"))
        _vector(ground.get("point_native"), "point_native")
    elif ground.get("mode") == "camera_up_constrained_ransac":
        for key in ("max_up_deviation_degrees", "plane_tolerance_m", "min_inlier_fraction"):
            _positive(ground, key)
        if ground["max_up_deviation_degrees"] >= 90 or ground["min_inlier_fraction"] > 1:
            raise ValueError("Invalid ground angle or inlier fraction")
        _positive(ground, "max_camera_below_plane_m", zero=True)
        for key, minimum, maximum in (("iterations", 1, 5000), ("max_fit_points", 3, 50000), ("min_inliers", 3, 50000)):
            if type(ground.get(key)) is not int or not minimum <= ground[key] <= maximum:
                raise ValueError("Invalid bounded ground setting: " + key)
        if type(ground.get("random_seed")) is not int or ground["random_seed"] < 0:
            raise ValueError("Ground random_seed must be a nonnegative integer")
    else:
        raise ValueError("Unknown ground estimation mode")
    return value


def _bind_reference(reference, source, name):
    recording = source.get("recording_id")
    if recording is None and source.get("run_dir"):
        recording = Path(source["run_dir"]).parent.name
    if (reference["recording_id"] != recording
            or reference["geometry_fingerprint"] != source.get("geometry_fingerprint")):
        raise ValueError(name + " belongs to another recording or geometry fingerprint")


def _level_reference_up(reference, plane, first_camera):
    explicit = "normal_native" in reference
    normal = _vector(reference["normal_native"] if explicit else plane["normal_native"], "level reference normal")
    point = _vector(reference["point_native"] if explicit else plane["point_native"], "level reference point")
    normal /= np.linalg.norm(normal)
    flipped = float((first_camera - point) @ normal) < 0
    if flipped:
        normal = -normal
    return normal, {"normal_native": normal.tolist(), "point_native": point.tolist(),
                    "source": "explicit_observed_reference_plane" if explicit else "fitted_observed_ground_plane",
                    "normal_flipped_toward_first_camera": flipped}


def _camera_height_scale(reference, plane, first_camera, up):
    normal = _vector(plane["normal_native"], "scale reference plane normal")
    normal /= np.linalg.norm(normal)
    point = _vector(plane["point_native"], "scale reference plane point")
    denominator = float(normal @ up)
    if abs(denominator) <= 1e-8:
        raise ValueError("Camera-height ray must intersect the supporting plane along positive up")
    if denominator < 0:
        normal, denominator = -normal, -denominator
    # A plane centroid need not be under the camera. Solve the vertical-ray
    # intersection rather than projecting camera-minus-centroid onto up.
    height_native = float(normal @ (first_camera - point)) / denominator
    if not math.isfinite(height_native) or height_native <= 1e-6:
        raise ValueError("First camera must have a positive resolved height above the reference plane")
    factor = reference["height_metres"] / height_native
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Derived research scale must be finite and positive")
    return factor, {"availability": "applied", "reference": deepcopy(reference),
                    "method": "first_camera_up_ray_intersection_with_observed_ground_plane",
                    "first_camera_native": first_camera.tolist(), "up_native": up.tolist(),
                    "plane_normal_native": normal.tolist(), "plane_point_native": point.tolist(),
                    "first_camera_height_native": height_native, "camera_height_metres": reference["height_metres"],
                    "metres_per_native_unit": factor, "geometry_scale_verified": False,
                    "height_is_assumed": reference["declaration"] == "user_assumed_camera_height",
                    "gravity_measured": False}


def estimate_ground(points, camera_centers, rotations, assumptions, *, prior_up=None):
    """Fit a constrained observed plane; gravity and the floor label stay assumed."""
    settings = assumptions["ground_estimation"]
    if settings["mode"] == "explicit_plane":
        normal = _vector(settings["normal_native"], "normal_native")
        normal /= np.linalg.norm(normal)
        return {"normal_native": normal.tolist(), "point_native": settings["point_native"],
                "method": "explicit_user_assumed_plane", "assumption": True}
    # OpenCV camera y points down. Assuming the first camera is approximately
    # upright supplies a prior, not a measured gravity vector.
    prior = -np.asarray(rotations[0], float)[1] if prior_up is None else np.asarray(prior_up, float).copy()
    prior /= np.linalg.norm(prior)
    rng = np.random.default_rng(settings["random_seed"])
    sample = points[rng.choice(len(points), min(len(points), settings["max_fit_points"]), replace=False)]
    tolerance = settings["plane_tolerance_m"] / assumptions["metres_per_native_unit"]
    cosine = math.cos(math.radians(settings["max_up_deviation_degrees"]))
    minimum = max(settings["min_inliers"], math.ceil(len(sample) * settings["min_inlier_fraction"]))
    best = None
    for _ in range(settings["iterations"]):
        triangle = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if np.dot(normal, prior) < 0:
            normal = -normal
        if np.dot(normal, prior) < cosine:
            continue
        offset = float(triangle[0] @ normal)
        # A plane above the camera track is not called supporting floor.
        if np.min(camera_centers @ normal - offset) < -settings["max_camera_below_plane_m"] / assumptions["metres_per_native_unit"]:
            continue
        mask = np.abs(sample @ normal - offset) <= tolerance
        count = int(np.count_nonzero(mask))
        rank = (count, -offset / max(float(normal @ prior), 1e-12))
        if count >= minimum and (best is None or rank > best[0]):
            best = (rank, sample[mask])
    if best is None:
        return None
    inliers = best[1]
    point = np.mean(inliers, axis=0)
    _, spread, vt = np.linalg.svd(inliers - point, full_matrices=False)
    normal = vt[-1]
    if normal @ prior < 0:
        normal = -normal
    if normal @ prior < cosine or len(spread) < 2 or spread[1] < 1e-8:
        return None
    if np.min((camera_centers - point) @ normal) < -settings["max_camera_below_plane_m"] / assumptions["metres_per_native_unit"]:
        return None
    return {"normal_native": normal.tolist(), "point_native": point.tolist(),
            "method": "camera_up_prior_constrained_observed_plane_ransac", "assumption": True,
            "camera_up_prior_native": prior.tolist(), "sample_points": len(sample),
            "plane_inlier_count": len(inliers), "plane_inlier_fraction": len(inliers) / len(sample),
            "rms_native": float(np.sqrt(np.mean(((inliers - point) @ normal) ** 2))),
            "floor_identity_verified": False, "gravity_measured": False}


def _research_settings(config, units, up, scale):
    """Research-only validator used solely in an isolated numerical function."""
    if units != "assumed_metres" or scale.get("assumed") is not True or config.get("research_illustration") is not True:
        raise ValueError("Research costmap requires explicitly assumed coordinates")
    robot, planning = config["robot"], config["planning"]
    profile = {key: float(robot[key]) for key in (
        "footprint_radius", "height", "clearance", "max_slope_degrees", "max_step", "max_roughness")}
    profile["min_support_points"] = robot["min_support_points"]
    return (planning, robot, profile, common._basis(up), np.asarray(planning["origin"], float),
            float(planning["resolution"]), tuple(planning["shape"]), float(planning["support_height"]))


def research_costmap(points_native, plane, assumptions, *, up_native=None,
                     voxels_native=None, voxel_size_native=None):
    """Reuse common feasibility math without claiming verified scale or gravity."""
    factor = assumptions["metres_per_native_unit"]
    points = np.asarray(points_native, float) * factor
    reference_up = plane["normal_native"] if up_native is None else up_native
    basis = common._basis(reference_up)
    projected = points @ basis.T
    resolution = assumptions["grid_resolution_m"]
    origin = np.floor(np.min(projected[:, :2], axis=0) / resolution) * resolution - resolution
    dimensions = np.ceil((np.max(projected[:, :2], axis=0) - origin) / resolution).astype(int) + 1
    if int(dimensions[0]) * int(dimensions[1]) > assumptions["max_grid_cells"]:
        raise ValueError("Observed geometry exceeds the explicit research grid size bound")
    config = {"research_illustration": True, "robot": deepcopy(assumptions["robot"]),
              "planning": {"origin": origin.tolist(), "resolution": resolution,
                           "shape": [int(dimensions[1]), int(dimensions[0])],
                           "support_height": float(np.asarray(plane["point_native"]) @ basis[2] * factor),
                           "terrain": deepcopy(assumptions.get("terrain", {}))}}
    # Clone the function's namespace rather than monkeypatching the common
    # module. Its conservative surface, slope, footprint and unknown checks
    # are identical; only the provenance gate accepts this explicit demo.
    namespace = dict(common.build_costmap.__globals__, _settings=_research_settings)
    calculate = FunctionType(common.build_costmap.__code__, namespace,
                             "research_only_costmap", common.build_costmap.__defaults__)
    calculate.__kwdefaults__ = common.build_costmap.__kwdefaults__
    terrain_voxels = None if voxels_native is None else {
        "centers": np.asarray(voxels_native["centers"], float) * factor}
    arrays, metadata = calculate(points, config, units="assumed_metres", up=reference_up,
                                scale={"assumed": True, "metres_per_native_unit": factor},
                                terrain_voxels=terrain_voxels,
                                voxel_size=None if voxel_size_native is None else voxel_size_native * factor)
    metadata.update(units="assumed_metres", research_illustration=True,
                    safety_validated=False, calibration_verified=False)
    return arrays, metadata


def _cell_point(cell, arrays):
    y, x = cell
    projected = np.r_[arrays["origin"] + np.array([x + .5, y + .5]) * arrays["resolution"][0],
                      arrays["support_height"][y, x]]
    return projected @ arrays["projection_basis"]


def _camera_endpoint(center, arrays):
    projected = center @ arrays["projection_basis"].T
    x, y = np.floor((projected[:2] - arrays["origin"]) / arrays["resolution"][0]).astype(int)
    shape = arrays["decision_state"].shape
    if not (0 <= y < shape[0] and 0 <= x < shape[1]) or not np.isfinite(arrays["support_height"][y, x]):
        return None
    projected[2] = arrays["support_height"][y, x]
    return projected @ arrays["projection_basis"]


def _demonstration_endpoints(arrays, minimum_separation):
    state = arrays["decision_state"]
    unseen = set(map(tuple, np.argwhere(state == common.TRAVERSABLE)))
    components = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        queue, component = deque([first]), {first}
        while queue:
            y, x = queue.popleft()
            for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                neighbor = (y + dy, x + dx)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    if not components:
        return None, 0
    component = min(components, key=lambda group: (-len(group), min(group)))
    def farthest(start):
        queue, distance = deque([start]), {start: 0}
        while queue:
            y, x = queue.popleft()
            for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                neighbor = (y + dy, x + dx)
                if neighbor in component and neighbor not in distance:
                    distance[neighbor] = distance[(y, x)] + 1
                    queue.append(neighbor)
        return min(distance, key=lambda cell: (-distance[cell], cell))
    start = farthest(min(component)); goal = farthest(start)
    endpoints = (_cell_point(start, arrays), _cell_point(goal, arrays))
    if np.linalg.norm(endpoints[1] - endpoints[0]) < minimum_separation:
        return None, len(component)
    return endpoints, len(component)


def plan_illustration(points_native, camera_centers_native, rotations, assumptions, source=None,
                      *, voxels_native=None, voxel_size_native=None):
    assumptions = validate_assumptions(assumptions)
    points = np.asarray(points_native, float)
    cameras = np.asarray(camera_centers_native, float)
    rotations = np.asarray(rotations, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Research surface points must be finite [N,3]")
    if cameras.ndim != 2 or cameras.shape[1:] != (3,) or len(cameras) < 1 or not np.isfinite(cameras).all():
        raise ValueError("A separate finite camera trajectory [S,3] is required")
    if rotations.shape != (len(cameras), 3, 3) or not np.isfinite(rotations).all():
        raise ValueError("Camera world-to-camera rotations must match the trajectory")
    result = {"schema_version": 1, "artifact_kind": "research_illustration_plan",
              "research_illustration": True, "label": LABEL, "source": deepcopy(source or {}),
              "planning_basis": "geometry_only", "semantic_guidance": False,
              "assumptions": assumptions, "status": "blocked_inputs", "reason": None,
              "path_points": [], "path_coordinate_units": "original_native_map_units",
              "assumed_up_vector": None, "ground_plane": None, "endpoints": None,
              "camera_trajectory_points": cameras.tolist(), "camera_trajectory_is_planned_route": False,
              "frame_scope": "final_cumulative_map_posthoc", "safety_validated": False,
              "clearance_certified": False, "calibration_verified": False,
              "limitations": common.LIMITATIONS + [
                  "Scale, gravity, floor identity and robot dimensions are research assumptions, not measurements.",
                  "A final-map illustration uses future observations and cannot establish online navigation or runtime latency.",
                  ("Forward mission endpoints use first and last observed floor samples with a bounded support adjustment."
                   if assumptions.get('endpoint_policy')=='recorded_forward_corridor' else
                   "Automatic demonstration endpoints are illustrative supported cells, not a requested robot mission."),
                  "No independent reference or accuracy claim; SAM surface labels are not a safety certificate."]}
    references = {key: assumptions[key] for key in ("level_reference", "camera_height_reference") if key in assumptions}
    for name, reference in references.items():
        _bind_reference(reference, result["source"], name)
    if references:
        result["input_assumptions"] = deepcopy(assumptions)
        result["research_calibration"] = {name: {"availability": "unavailable", "reference": deepcopy(reference),
                                                "reason": "No observed supporting plane has been established"}
                                          for name, reference in references.items()}
    # A terrain plane provides a support anchor, never a gravity measurement.
    # In this explicitly assumed illustration the upright first camera supplies
    # up independently, so an inclined floor keeps its nonzero slope.
    reference_up = -rotations[0, 1].copy()
    if np.linalg.norm(reference_up) < 1e-12:
        raise ValueError("First camera up axis must be nonzero")
    reference_up /= np.linalg.norm(reference_up)
    phone_up = reference_up.copy()
    if "up_vector_native" in assumptions:
        reference_up = _vector(assumptions["up_vector_native"], "up_vector_native")
        if np.linalg.norm(reference_up) < 1e-12:
            raise ValueError("up_vector_native must be nonzero")
        reference_up /= np.linalg.norm(reference_up)
    level = references.get("level_reference")
    explicit_level = level is not None and "normal_native" in level
    if explicit_level:
        reference_up, level_plane = _level_reference_up(level, None, cameras[0])
    result.update(assumed_up_vector=reference_up.tolist(),
                  slope_reference={"up_native": reference_up.tolist(), "gravity_measured": False,
                                   "source": "user_declared_level_reference" if explicit_level else (
                                       "explicit_assumed_up_vector" if "up_vector_native" in assumptions else "assumed_upright_first_camera")})
    if len(points) < 3:
        result["reason"] = "Insufficient observed geometry for an assumed ground plane"
        return result, None
    plane = estimate_ground(points, cameras, rotations, assumptions,
                            prior_up=reference_up if explicit_level else None)
    if plane is None:
        result["reason"] = "No sufficiently supported observed plane compatible with the assumed camera-up prior"
        return result, None
    result["ground_plane"] = plane
    if level is not None:
        if not explicit_level:
            reference_up, level_plane = _level_reference_up(level, plane, cameras[0])
        result["assumed_up_vector"] = reference_up.tolist()
        result["slope_reference"] = {"up_native": reference_up.tolist(), "gravity_measured": False,
                                     "source": "user_declared_level_reference"}
        result["research_calibration"]["level_reference"] = {
            "availability": "applied", "reference": deepcopy(level), "reference_plane": level_plane,
            "method": "observed_reference_plane_user_declared_level", "phone_up_original_native": phone_up.tolist(),
            "up_native": reference_up.tolist(),
            "angular_correction_degrees": math.degrees(math.acos(float(np.clip(phone_up @ reference_up, -1., 1.)))),
            "declared_reference_slope_degrees": 0., "declared_reference_grade_percent": 0.,
            "local_terrain_measurements_forced_flat": False,
            "gravity_measured": False, "geometry_up_verified": False,
            "fitted_ground_plane_evidence": deepcopy(plane),
        }
    height_reference = references.get("camera_height_reference")
    if height_reference is not None:
        try:
            factor, calibration = _camera_height_scale(height_reference, plane, cameras[0], reference_up)
        except ValueError as error:
            result["reason"] = str(error)
            result["research_calibration"]["camera_height_reference"]["reason"] = str(error)
            return result, None
        calibration["previous_metres_per_native_unit_assumption"] = assumptions["metres_per_native_unit"]
        assumptions["metres_per_native_unit"] = factor
        result["research_calibration"]["camera_height_reference"] = calibration
    try:
        arrays, metadata = research_costmap(points, plane, assumptions, up_native=reference_up,
                                            voxels_native=voxels_native, voxel_size_native=voxel_size_native)
    except ValueError as error:
        result["reason"] = str(error)
        return result, None
    factor = assumptions["metres_per_native_unit"]
    if assumptions['endpoint_policy'] == 'recorded_forward_corridor':
        from pipeline_common.research_route import forward_corridor_route
        targets=result['source'].get('forward_mission_target_points_native')
        route=forward_corridor_route(arrays,metadata,cameras*factor,rotations,assumptions['forward_mission'],
            target_points=None if targets is None else np.asarray(targets)*factor)
        result.update(status=route['status'],reason=route['reason'],
                      mission=route['mission'],
                      path_points=(np.asarray(route['path']).reshape(-1,3)/factor).tolist(),
                      path_cells=route['path_cells'],waypoint_cells=route.get('waypoint_cells',[]),
                      swept_footprint_checked=route.get('swept_footprint_checked',False))
        if route['status']=='ok':
            result['endpoints']={'selection':'requested_recorded_forward_corridor','assumed':True,
                                 'start_native':(np.asarray(route['start'])/factor).tolist(),
                                 'goal_native':(np.asarray(route['goal'])/factor).tolist()}
        states=arrays['decision_state']
        result['diagnostics']={'surface_points':len(points),'grid_shape':list(states.shape),
            'unknown_cells':int(np.count_nonzero(states==common.UNKNOWN)),
            'blocked_cells':int(np.count_nonzero(states==common.BLOCKED)),
            'traversable_cells':int(np.count_nonzero(states==common.TRAVERSABLE)),
            'planning_check_metadata':metadata}
        return result,arrays
    endpoints = (_camera_endpoint(cameras[0] * factor, arrays),
                 _camera_endpoint(cameras[-1] * factor, arrays))
    map_frame = result["source"].get("map_frame", "research_native_world")
    camera_reason = "Camera projections lack observed supporting ground"
    route = None
    policy = assumptions["endpoint_policy"]
    if policy != "largest_observed_component" and all(point is not None for point in endpoints):
        route = common.plan_requests(arrays, metadata, [{"start": endpoints[0].tolist(), "goal": endpoints[1].tolist()}], map_frame)[0]
        camera_reason = route["reason"]
        if route["status"] == "ok" and np.linalg.norm(endpoints[1] - endpoints[0]) < assumptions["minimum_endpoint_separation_m"]:
            route = None
            camera_reason = "Camera projected endpoints do not meet the assumed minimum separation"
    endpoint_kind = "camera_first_last_xy_projected_to_observed_support"
    component_cells = None
    if policy != "camera_projected" and (route is None or route["status"] != "ok"):
        endpoints, component_cells = _demonstration_endpoints(arrays, assumptions["minimum_endpoint_separation_m"])
        endpoint_kind = "automatic_demonstration_endpoints_on_largest_observed_traversable_component"
        if endpoints is not None:
            route = common.plan_requests(arrays, metadata, [{"start": endpoints[0].tolist(), "goal": endpoints[1].tolist()}], map_frame)[0]
        else:
            route = None
    if route is not None:
        result.update(status=route["status"], reason=route["reason"],
                      path_points=(np.asarray(route["path"]).reshape(-1, 3) / factor).tolist(),
                      endpoints={"selection": endpoint_kind, "assumed": True,
                                 "start_native": (endpoints[0] / factor).tolist(),
                                 "goal_native": (endpoints[1] / factor).tolist()},
                      path_cells=route.get("path_cells", []), path_cost_assumed_metres=route.get("path_cost"))
    else:
        reason = camera_reason if policy == "camera_projected" else (
            "No observed traversable component meets the declared endpoint separation; unknown and obstacle cells remain excluded")
        result.update(status="no_path", reason=reason)
    states = arrays["decision_state"]
    result["diagnostics"] = {
        "surface_points": len(points), "camera_endpoint_attempt_reason": camera_reason,
        "largest_component_cells": component_cells, "grid_shape": list(states.shape),
        "unknown_cells": int(np.count_nonzero(states == common.UNKNOWN)),
        "blocked_cells": int(np.count_nonzero(states == common.BLOCKED)),
        "traversable_cells": int(np.count_nonzero(states == common.TRAVERSABLE)),
        "planning_check_metadata": metadata,
    }
    return result, arrays


def export_research_plan(run_dir, output, assumptions, geometry_archive=None):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output == run_dir or output.is_relative_to(run_dir):
        raise ValueError("Research output must be separate from the original run")
    if output.exists():
        raise FileExistsError("Research output already exists; choose a new separate folder")
    run, geometry = _read(run_dir / "run.json"), _read(run_dir / "geometry/manifest.json")
    if run.get("status") != "complete" or geometry.get("status") != "complete":
        raise ValueError("Research export requires a completed immutable source run")
    if run.get("geometry_identity", {}).get("geometry_fingerprint") != geometry.get("geometry_fingerprint"):
        raise ValueError("Run and geometry fingerprint differ")
    sequence_identity = run.get("sequence", {})
    recording_id = sequence_identity.get("sequence_id")
    if recording_id:
        if sequence_identity.get("manifest_digest") != geometry.get("sequence_digest"):
            raise ValueError("Run sequence identity and geometry sequence digest differ")
    elif geometry.get("derivation"):
        raise ValueError("A derived run requires its immutable source sequence identity")
    else:
        recording_id = run_dir.parent.name
    archive = Path(geometry_archive).resolve() if geometry_archive else run_dir / "geometry/cache/geometry.npz"
    if not archive.is_file() and geometry_archive is None:
        archive = Path(geometry.get("cache_path", "")) / "geometry.npz"
    archive_sha = file_sha256(archive)
    if archive_sha != geometry.get("archive_sha256"):
        raise ValueError("Source geometry archive hash differs from the original manifest")
    with np.load(archive, allow_pickle=False) as saved:
        numeric = {key: saved[key] for key in GEOMETRY_KEYS}
    validate_geometry(numeric, allow_single_frame=True)
    if geometry_fingerprint(numeric) != geometry["geometry_fingerprint"]:
        raise ValueError("Source numeric geometry fingerprint differs")
    points, depth, confidence = numeric["world_points"], numeric["depth"], numeric["world_points_conf"]
    depth = depth.reshape(points.shape[:-1])
    valid = np.isfinite(points).all(-1) & np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    valid &= (confidence > 0) & (confidence >= geometry.get("settings", {}).get("min_confidence", 1.5))
    for index, transform in enumerate(geometry.get("transforms", [])):
        left, top, right, bottom = transform.get("pad_ltrb", [0, 0, 0, 0])
        if top: valid[index, :top] = False
        if bottom: valid[index, -bottom:] = False
        if left: valid[index, :, :left] = False
        if right: valid[index, :, -right:] = False
    extrinsic = numeric["extrinsic"]
    rotations = extrinsic[:, :, :3]
    cameras = -np.einsum("sji,sj->si", rotations, extrinsic[:, :, 3])
    source = {"run_dir": str(run_dir), "run_json_sha256": file_sha256(run_dir / "run.json"),
              "recording_id": recording_id,
              "geometry_manifest_sha256": file_sha256(run_dir / "geometry/manifest.json"),
              "geometry_archive": str(archive), "archive_sha256": archive_sha,
              **{key: geometry.get(key) for key in ("geometry_fingerprint", "input_fingerprint", "processed_grid_id", "map_frame", "units", "pose_revision")},
              "original_planning_status": run.get("planning_status"),
              "original_scale": geometry.get("scale"), "original_up": geometry.get("up"),
              "source_fixture": run.get("fixture"), "geometry_point_source": "saved reconstructed world surface points",
              "helper_sha256": file_sha256(__file__), "common_planning_sha256": file_sha256(common.__file__)}
    if geometry.get('derivation'):
        source['derived_geometry'] = deepcopy(geometry['derivation'])
    if assumptions.get('endpoint_policy')=='recorded_forward_corridor':
        # A target in each actual view remains meaningful when camera pitch
        # changes; fixed horizontal lookahead can fall outside the saved crop.
        targets=[];pixels=[]
        for index in (0,len(points)-1):
            h,w=points[index].shape[:2];x=w//2;y=min(h-2,int(.90*h))
            patch=points[index,y-1:y+2,x-1:x+2]
            accepted=valid[index,y-1:y+2,x-1:x+2]
            if np.count_nonzero(accepted)<3:raise ValueError('Observed forward-view mission target lacks saved depth/confidence')
            targets.append(np.median(patch[accepted],axis=0).tolist());pixels.append([x,y])
        source['forward_mission_target_points_native']=targets
        source['forward_mission_target_processed_pixels']=pixels
        source['derived_geometry']=geometry.get('derivation')
    map_manifest = run_dir / "map/manifest.json"
    voxels_native, voxel_size_native = None, None
    if map_manifest.is_file():
        saved_map = _read(map_manifest)
        if saved_map.get("geometry_fingerprint") != geometry["geometry_fingerprint"]:
            raise ValueError("Source voxel map and geometry identities differ")
        source.update(voxel_map_manifest_sha256=file_sha256(map_manifest),
                      voxel_size_native=saved_map.get("voxel_size"), voxel_origin_native=saved_map.get("origin"))
        voxel_path = run_dir / "map/voxels.npz"
        if voxel_path.is_file():
            with np.load(voxel_path, allow_pickle=False) as saved:
                indices, centers = saved["voxel_indices"], saved["centers"]
            voxel_size_native = saved_map["voxel_size"]
            expected = np.asarray(saved_map["origin"]) + (indices + .5) * voxel_size_native
            if centers.shape != indices.shape or not np.allclose(centers, expected, rtol=0, atol=1e-9):
                raise ValueError("Saved voxel centers differ from their map indices/origin/size")
            voxels_native = {"centers": centers}
            source.update(voxel_map_sha256=file_sha256(voxel_path),
                          slope_point_source="saved_fused_LingBot_map_voxel_centers",
                          terrain_estimator_sha256=file_sha256(Path(common.__file__).with_name("terrain.py")))
    if assumptions.get("require_voxel_terrain") and voxels_native is None:
        raise ValueError("Requested voxel terrain requires saved map/voxels.npz")
    result, arrays = plan_illustration(points[valid], cameras, rotations, assumptions, source,
                                       voxels_native=voxels_native, voxel_size_native=voxel_size_native)
    if geometry.get('derivation', {}).get('planning_admitted') is False:
        result.update(status='blocked_inputs',
                      reason='Saved pose interpretation is unverified; this copy is for display only',
                      path_points=[], path_cells=[], endpoints=None,
                      planning_admitted=False)
        arrays = None
    output.mkdir(parents=True, exist_ok=False)
    (output / "assumptions.json").write_text(json.dumps(result["assumptions"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if arrays is not None:
        np.savez_compressed(output / "research_costmap.npz", **arrays)
        result["costmap_file"] = "research_costmap.npz"
        result["costmap_sha256"] = file_sha256(output / "research_costmap.npz")
    path = output / "research_plan.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assumptions", type=Path, required=True, help="Explicit research assumption JSON; see DEFAULT_ASSUMPTIONS")
    parser.add_argument("--geometry-archive", type=Path, help="Existing numeric archive when extracted outside the original cache path")
    args = parser.parse_args(argv)
    result, path = export_research_plan(args.run, args.output, _read(args.assumptions), args.geometry_archive)
    print(json.dumps({"research_plan": str(path), "status": result["status"], "reason": result["reason"], "label": LABEL}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
