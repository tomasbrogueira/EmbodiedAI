# Path planning

Plan directly on reconstructed 3D point rows. Coordinates, row order, colours, masks, slopes and video poses remain aligned. No voxel conversion, surface filling or route smoothing is used. Distances are reconstruction units, not metres.

## Layout

| Files | Purpose |
| --- | --- |
| slope.py, floor_reference.py, video_reference.py | Relative inclination and automatic floor reference |
| cost_evidence.py, cost_map.py | Row correspondence, costs and valid connections |
| planner.py, recording_endpoints.py | Exact Dijkstra and first/last-pose endpoints |
| dataset_reference.py, dataset_prepare.py, dataset_path.py | Recording adapters, frozen calibration and certified native search |
| dataset_inventory.py, dataset_gpu.py, dataset_cpu.py, dataset_resources.py | Resumable batch work and bounded resource sharing |
| generate_masks.py, sam_cpu_video.py | Existing SAM masks with video storage on CPU |
| corridor_astar_feasible.cpp | Accelerated search checked against the exact CostMap |
| visualization/ | Viser viewer, overview and isolated route rendering |
| test_*.py | Synthetic, cache and scheduling regressions |

Repository-root reconstruct.py initializes streaming LingBot from eight uniformly spread full-recording views, then processes remaining frames and restores all output fields to original frame order. This is an offline recording method. Model code, weights, environments and generated outputs stay outside Git.

## Contracts

**Slope:** estimate_slope(points, n_ref, config, frame_ids=..., return_quality=...) preserves every row. It fits up to 128 distinct neighbours from the same source frame within the automatic radius, using PCA and one Huber refit. Fewer than six effective samples or an undefined normal gives NaN. Quality is diagnostic, not a feasibility threshold. Angles are radians relative to the estimated ground, not measured gravity.

**Reference:** video_reference checks the first five saved frames. The dataset adapter tries subsequent five-frame windows only if needed and records the accepted window. Mask erosion, deterministic RANSAC and frame-agreement rules are unchanged. Missing reference means no route; there is no invented plane fallback. Use floor masks indoors and ground outdoors.

**Costs:** build_cost_map(points, slope, n_ref, config) provides aligned costs and cost_map.neighbours(id). Unknown or infeasible rows cost infinity. The unchanged finite cost is:

    1 + w_slope * (slope / max_slope)^2
      + w_clearance * exp(-min(nearest_obstacle_distance, clearance_scale) / clearance_scale)

For automatic radius r, defaults are max_slope=pi/6, weights 1, at least six support positions, clearance_scale=r, connection_radius=r/2, max_gap=r/4, obstacle_margin=r/20 and generic clearance_height=r. Nodes and whole edges must clear observed non-floor geometry in the region swept upward along n_ref. Connections also require grade and continuous support. Coincident feasible rows retain zero-length edges. Finite costs and absent observations do not establish physical body clearance.

The recording adapter freezes **H = max(r, median positive signed camera-to-ground drop)** from baseline-valid beneath-camera associations before additional clearance filtering. Visible-ground fallback never contributes. Provenance retains H and contributing frame/row IDs. Missing calibration means no route. This relative camera-height assumption is not a metric body guarantee.

**Dijkstra:** find_path(points, cost_map, start_id, goal_id) uses existing costs and neighbours. Edge weight is distance(a,b) * (cost[a]+cost[b])/2. IDs are original point rows. Equal valid endpoints return one row and zero cost. invalid_start, invalid_goal and no_path return empty arrays and infinite cost.

**Recording endpoints:** find_recording_path verifies row/pose correspondence and associates the actual first and last poses with observed support beneath the camera within r/2 laterally. Blocked nearest support fails instead of being skipped. The dataset adapter additionally allows same-frame visible ground only when beneath-camera support is missing. It fixes the earliest/latest accepted poses and reports omitted frames. On disconnection it keeps the start and returns the reachable original row nearest in 3D to the intended goal (exact ties: lowest row ID), labelled **partial; intended goal unreachable**. Generic Dijkstra failure behavior is unchanged. Native candidates, IDs and edge weights remain certified against the exact CostMap.

## Run

Core code needs NumPy and SciPy; visualization also needs Pillow and Viser 1.1.1. requirements.txt lists these existing dependencies. Linux/KTH batch workers use the existing .setup/envs/{lingbot-map,sam3,orchestrator-visualizer} environments. Reconstruction and masks use their installed model environments. Keep outputs outside the repository and BLAS threads bounded.

From the repository root, with a new external batch directory:

    R="$PWD"
    B=/absolute/path/outside/repository/new-batch
    P="$R/.setup/envs/orchestrator-visualizer/bin/python"
    export PYTHONPATH="$R/path-planning:$R/orchestrator"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    "$P" path-planning/dataset_inventory.py --repo "$R" --output "$B"
    g++ -O3 -std=c++17 -ffp-contract=off -shared -fPIC path-planning/corridor_astar_feasible.cpp -o path-planning/corridor_astar_feasible.so

Run requested jobs in your allocated session (e.g. tmux). GPU and CPU workers may overlap under the existing allocation lock. Recordings over 900 frames remain exclusive, with at most one preparation thread alongside route search:

    "$P" path-planning/dataset_gpu.py --manifest "$B/manifest.json"
    "$P" path-planning/dataset_cpu.py --manifest "$B/manifest.json"

Or process one existing recording run explicitly:

    "$P" path-planning/dataset_prepare.py --repo "$R" --output "$B/runs/indoor/VIDEO"
    "$P" path-planning/dataset_path.py --repo "$R" --output "$B/runs/indoor/VIDEO"

View results and optionally generate one overview:

    "$P" -m visualization.dataset_viewer --manifest "$B/manifest.json" --host 127.0.0.1 --port 8094
    "$P" -m visualization.dataset_status --manifest "$B/manifest.json" --once

Workers accept any manifest filename. Locks, receipts and reports live beside the manifest. The overview accepts --viewer-url and refreshes every 20 seconds without --once. Stop workers before refreshing inventory; optional --pull only fast-forwards the checkout. Computational failures remain errors; completed no-route outcomes remain explicit.

## Cache and visualization

Inputs, filters, row correspondence and source hashes must match. Changed inputs/settings preserve existing runs and require a fresh output directory. Consolidating adapters changes source identity: old artifacts are **not recertified** by this cleanup. Preparation refuses mismatched geometry; the viewer withholds stale costs/routes. Existing viewers can retain their original code and manifests for historical verified results. Never edit receipts to disguise mismatches.

The single **Colour** selector defaults to **Cost map**, with **Source / Cost map / Steepness**. Finite costs use blue–cyan–yellow–red: red is expensive but traversable, gray unavailable. Steepness reads original slope.npy rows on a fixed 0–90 degree scale, with unknown values gray. Switching colours updates only the cloud and legend, preserving camera and paths.

Camera trajectory is cyan. Ground routes are magenta with a white outline at actual coordinates; partial endpoints are orange and unreachable intended goals red. Display sampling never reduces the planning graph. The isolated client in visualization/viewer-client/ is generated and ignored. Its overlay supports Viser 1.1.1 and fails closed if that client changes. Save current view writes beside the selected recording.

## Tests

Compile the native library above, then run:

    export PYTHONPATH="$PWD/path-planning:$PWD/orchestrator"
    export DATASET_REPO="$PWD"
    export DATASET_NATIVE_LIBRARY="$PWD/path-planning/corridor_astar_feasible.so"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -s path-planning -v

Reconstruction wrapper tests use CPU tensors and skip if PyTorch is absent; they never load weights or run model inference. Tests cover row alignment, weighted optimal paths, coincident points, disconnection and fixed-start partial routes, calibration, native/Python parity, stale caches, scheduling and all three viewer modes.
