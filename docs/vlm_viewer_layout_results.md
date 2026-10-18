# Wider overview and camera following

The first-person following described below was subsequently replaced by
third-person following with adjustable distance, height and an 80° default
field of view. See [third-person browser checks](../data/vlm_maps/live_demo/third_person_verification.json)
and [offline frame-count mode](../plugins/small_obstacles/STREAMING.md#offline-frame-count-mode)
for the current behavior. The validation below records the earlier layout test.

Validated on 2026-10-11 in the optional `vlm_improvements` viewer. The live service
was restarted on port 8080 and left running with LingBot, Qwen and SAM resident.
No geometry inference, segmentation, frame sampling or map-projection settings
were changed.

## Layout

The right panel is docked at 520 pixels, with Overview, Prompts and Settings tabs.
Overview contains video selection, mode selection, Start/Stop, Follow camera,
compact live metrics, current queries, camera pose and adjacent RGB/mask previews.
Prompt editing and model unloading have their own tabs. Metrics include elapsed
time, first-map latency, geometry and map rates/counts, point counts, GPU readings
and VLM calls. Frames are reduced only for the display; saved RGB and masks retain
their original data.

Real browser checks through the authenticated JupyterHub proxy verified that
the overview and previews fit without vertical scrolling at 1440×900 and
1280×720. The 720p layout uses tighter metric spacing and 90-pixel previews.
The overview remains resizable by dragging its panel edge. Large custom text or
unusually long error messages can increase its height beyond these tested cases.

## Following the camera

Follow camera is enabled by default. Each new run initializes the viewport from
the reconstructed camera pose rather than a generic external map viewpoint.
As geometry advances, the viewer follows camera translation, orientation,
including roll, and the predicted vertical field of view. The transform is in
the original map coordinates: OpenCV +Z forward and -Y up. Position is assigned
before look-at to account for Viser translating its target with its position.
Near clipping scales with scene extent in model units.

Disable Follow camera to navigate freely. Re-enabling it returns to the latest
camera view, including when a completed video remains on the same final frame.
The camera marker is hidden while viewing from its own origin, so its axes,
frustum and label do not obscure the map. Free inspection shows that marker and
the cyan trajectory. Following modifies only the viewport, never scene geometry.

## Validation

All 38 plugin tests passed. The additional camera checks cover a rotated camera
with roll, position/look-at ordering, pause/resume, new clients, new video reset,
HTML escaping and paired frame previews. Browser checks validated free navigation
and resuming at a saved camera pose, along with access to Prompts and Settings.

A full live VLM run of `IMG_5516.mp4` then completed 50 geometry frames and 25 map
updates, with five Qwen calls and no dropped selected frames. Browser camera
telemetry matched 34 observed moving geometry poses. The final browser position,
forward direction and field of view matched the final estimated camera. The
ordinary overview still fit without scrolling at 1280×720 with all models loaded.
No browser JavaScript errors were reported. The output retained 3,400,657 points,
including 124,895 obstacle-marked points.

- [Layout and free-navigation checks](../data/vlm_maps/live_demo/viewer_layout_verification.json)
- [Live following checks and full status](../data/vlm_maps/live_demo/live_camera_follow_verification.json)
- [Live viewer at 720p](../data/vlm_maps/live_demo/live_camera_follow_viewer.png)
- [Saved-view overview at 900p](../data/vlm_maps/live_demo/viewer_layout_1440x900.png)
- Live run: `data/vlm_maps/live_demo/IMG_5516_81bdbcd150ef/`

Open `http://localhost:8080/` through the established access setup, or the existing
JupyterHub host's `/user/jorisk/proxy/8080/` route. Refresh an already open viewer
to load the new layout.
