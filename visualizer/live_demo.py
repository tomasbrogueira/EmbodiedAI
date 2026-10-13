#!/usr/bin/env python3
"""Start and watch an actual optional VLM mapping run from a Viser button."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4

import numpy as np
import viser

from orchestrator.paths import ROOT
sys.path.insert(0, str(ROOT))
from orchestrator.shared import read_rgb
from orchestrator.model_session import ModelSession
from vlm.propose import instruction_for
from vlm.prompt_presets import DEFAULT_NAME, DEFAULT_INSTRUCTION, PromptPresets, validate_instruction, parse_sam_prompts
from orchestrator.streaming import parse_args as stream_args, run
from visualizer.scene_pose import PoseOverlay, CameraFollower, FOLLOW_DISTANCE, FOLLOW_HEIGHT, FOLLOW_FOV
from visualizer.viewer_overview import overview_html, previews_html


def command(args, video, output, frame_count=0, map_step=2):
    return [args.lingbot_python, '-m', 'plugins.small_obstacles.run_streaming',
            '--video-path', str(video), '--output-dir', str(output),
            '--lingbot-python', args.lingbot_python, '--vlm-python', args.vlm_python,
            '--sam-python', args.sam_python,
            '--input-fps', '4', '--frame-count', str(frame_count), '--map-step', str(map_step), '--vlm-interval', '2',
            '--kv-cache-window', '8', '--worker-threads', '4',
            '--cuda-cache-policy', 'budget', '--sam-text-cache', '--save-frames']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos-dir', type=Path, default=ROOT/'data/videos')
    parser.add_argument('--output-root', type=Path, default=ROOT/'data/vlm_maps/live_demo')
    parser.add_argument('--port', type=int, default=8083)
    parser.add_argument('--initial-run', type=Path, help='Show an explicitly labeled saved map until a new run starts')
    parser.add_argument('--lingbot-python', default=('/opt/conda/envs/lingbot-map/bin/python'
        if Path('/opt/conda/envs/lingbot-map/bin/python').exists() else sys.executable))
    parser.add_argument('--vlm-python', default='/opt/conda/bin/python')
    parser.add_argument('--sam-python', default=('/opt/conda/envs/sam3/bin/python'
        if Path('/opt/conda/envs/sam3/bin/python').exists() else sys.executable))
    args = parser.parse_args()
    videos = {str(p.relative_to(args.videos_dir)): p.resolve()
              for p in sorted(args.videos_dir.rglob('*.mp4'))}
    if not videos:
        parser.error('No MP4 videos found')
    if args.initial_run and not (args.initial_run/'status.json').is_file():
        parser.error('Initial saved run must contain status.json')
    initial_config = {}
    if args.initial_run and (args.initial_run/'config.json').is_file():
        initial_config = json.loads((args.initial_run/'config.json').read_text())
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    server = viser.ViserServer(host='0.0.0.0', port=args.port, label='Live VLM → SAM → LingBot map')
    server.scene.set_up_direction('+z')
    server.gui.configure_theme(control_width='large', show_logo=False, show_share_button=False)
    server.gui.main_panel.dock_right()
    server.gui.main_panel.set_width(520)
    default = 'outdoor/IMG_5516.mp4' if 'outdoor/IMG_5516.mp4' in videos else next(iter(videos))
    for name, path in videos.items():
        if str(path) == initial_config.get('video_path'):
            default = name
            break
    presets = PromptPresets(args.output_root/'prompt_presets')
    saved_prompts = presets.list()
    tabs = server.gui.add_tab_group()
    with tabs.add_tab('Overview'):
        selection = server.gui.add_dropdown('Video', options=list(videos), initial_value=default)
        mode = server.gui.add_dropdown('Prompt source', options=['VLM proposals', 'Manual SAM prompts', 'Skip VLM'],
            initial_value='VLM proposals', hint='Edit the instruction or object names in the Prompts tab.')
        start = server.gui.add_button('Start processing', color='green')
        stop = server.gui.add_button('Stop processing', disabled=True)
        colour_mode = server.gui.add_dropdown('Colour', options=['Traversability'], initial_value='Traversability')
        route_visible = server.gui.add_checkbox('Show route', initial_value=True)
        legend = server.gui.add_markdown('Steepness/cost modes become available after finalized analysis.')
        follow_camera = server.gui.add_checkbox('Follow camera', True,
            hint='Third-person view behind and above the reconstructed camera. Adjust distance and field of view in Settings, or disable to inspect freely.')
        status = server.gui.add_html(overview_html({}, message='Choose a video. Edit the Prompts tab if needed, then Start processing.'))
        preview = server.gui.add_html('', visible=False)
    with tabs.add_tab('Prompts'):
        target = server.gui.add_dropdown('VLM detection target', options=['non_traversable', 'traversable'], initial_value='non_traversable')
        extra_enabled = server.gui.add_checkbox('Extra SAM traversability', initial_value=False)
        extra_editor = server.gui.add_text('Independent traversability queries', 'path', multiline=True)
        server.gui.add_markdown('Extra queries run alongside VLM or with it disabled. Non-traversable labels veto overlaps in the cost map.')
        sam_editor = server.gui.add_text('SAM object names', 'log', multiline=True, visible=False)
        sam_notice = server.gui.add_markdown('One concrete object name per line, or separate names with commas. '
            'Up to 8 names are queried on every map update. Qwen is skipped.', visible=False)
        with server.gui.add_folder('VLM prompt', expand_by_default=True) as vlm_folder:
            preset_selection = server.gui.add_dropdown('Saved prompt',
                options=[DEFAULT_NAME, *saved_prompts], initial_value=DEFAULT_NAME)
            prompt_editor = server.gui.add_text('VLM instruction', DEFAULT_INSTRUCTION, multiline=True)
            preset_name = server.gui.add_text('Save as', '')
            save_prompt = server.gui.add_button('Save prompt')
            prompt_notice = server.gui.add_markdown('Edit, then Start from Overview to test. Save a named preset to reuse it. '
                'Keep JSON with a `prompts` list of concrete object names for SAM.')
    with tabs.add_tab('Settings'):
        sam_enabled = server.gui.add_checkbox('SAM segmentation', initial_value=not initial_config.get('no_sam', False))
        support_enabled = server.gui.add_checkbox('Floor/ground support segmentation', initial_value=bool(initial_config.get('support_prompt')))
        slopes_enabled = server.gui.add_checkbox('Slope / steepness analysis', initial_value=initial_config.get('steepness', False))
        costs_enabled = server.gui.add_checkbox('Cost map', initial_value=initial_config.get('cost_map', False))
        planner_enabled = server.gui.add_checkbox('Path planning (requires costs)', initial_value=initial_config.get('plan', False))
        analysis_workers = server.gui.add_number('Analysis CPU workers (0 = automatic)', initial_value=initial_config.get('analysis_workers', 0), min=0, max=32, step=1)
        reference_source = server.gui.add_dropdown('Reference/calibration evidence', options=['base_support', 'traversability'], initial_value='base_support')
        server.gui.add_markdown('Analysis runs on the finalized map. Slope off + planning on uses an explicit slope-disabled policy. Visibility and computation are separate.')
        sampling = server.gui.add_dropdown('Processing mode',
            options=['Standard sampling (4 FPS)', 'Offline frame count'], initial_value='Standard sampling (4 FPS)',
            hint='Both modes build incrementally without dropping selected frames. Offline chooses a count over the full video.')
        frame_count = server.gui.add_number('Frames to process', initial_value=200, min=2, max=1024, step=1,
            disabled=True, hint='Unique source frames, evenly spaced from start to end. Capped at the number available in the video.')
        map_every = server.gui.add_dropdown('Map every', options=['2 selected frames (original)', 'Selected frame'],
            initial_value='2 selected frames (original)', hint='Every frame runs geometry. This sets how often SAM and projection update the map.')
        server.gui.add_markdown('Offline keeps the same incremental reconstruction and model settings. '
            'Choose a larger frame count for denser sampling; processing can take longer. '
            'Counts, elapsed time and map updates are shown in Overview. More frames do not guarantee better geometry.')
        follow_distance = server.gui.add_slider('Follow distance', min=.3, max=3., step=.05,
            initial_value=FOLLOW_DISTANCE, hint='Larger values move the viewpoint farther away. Relative to scene size.')
        follow_height = server.gui.add_slider('Follow height', min=0., max=2., step=.05,
            initial_value=FOLLOW_HEIGHT, hint='Raise or lower the viewpoint above the followed camera.')
        follow_fov = server.gui.add_slider('Field of view (°)', min=45., max=110., step=1.,
            initial_value=FOLLOW_FOV, hint='Larger angles show more of the scene and reduce the zoom.')
        unload = server.gui.add_button('Unload models / free GPU')
        server.gui.add_markdown('Models stay loaded between runs. **Unload models / free GPU** releases the session.\n\n'
            'The map accumulates observations. Red points show obstacle-mask observations. '
            'With **Follow camera** enabled, the viewer follows from behind and above the reconstructed camera. '
            'Increase **Follow distance** or **Field of view** for a wider view. '
            'Disable it to orbit the map; the orange camera marker and cyan trajectory show the estimated pose. '
            'Position is relative to the first camera and scale is in model units.\n\n'
            'Current model settings: cache 8, four camera iterations. Standard sampling uses approximately 4 input FPS; '
            'offline sampling uses your frame count across the whole video. Mapping defaults to every second selected frame, '
            'VLM refresh every 2 video seconds. Selected frames are processed as quickly as possible. '
            'The smaller cache changes geometry. The original pipeline remains available.')
    loaded_prompt = {'name': DEFAULT_NAME, 'instruction': DEFAULT_INSTRUCTION, 'loading': False}
    state = {'thread': None, 'directory': args.initial_run.resolve() if args.initial_run else None,
             'stop_requested': False, 'launch_error': None}
    # Viser invokes on_update synchronously when a callback assigns .value.
    # Saving a preset selects it, which enters load_prompt under this lock.
    lock = threading.RLock()
    cancel = threading.Event()
    session = ModelSession()
    pose_overlay = PoseOverlay(server)
    follower = CameraFollower(server)

    def update_prompt_controls(running=False):
        manual = mode.value == 'Manual SAM prompts'
        mode.disabled = target.disabled = running
        for control in (extra_enabled, extra_editor, sam_enabled, support_enabled, slopes_enabled, costs_enabled, planner_enabled, reference_source, analysis_workers):
            control.disabled = running
        sam_editor.visible = sam_notice.visible = manual
        vlm_folder.visible = mode.value == 'VLM proposals'
        target.visible = mode.value == 'VLM proposals'
        sam_editor.disabled = running
        prompt_editor.disabled = preset_selection.disabled = preset_name.disabled = save_prompt.disabled = running or manual
        sampling.disabled = map_every.disabled = running
        frame_count.disabled = running or sampling.value != 'Offline frame count'

    @sampling.on_update
    def change_sampling(_):
        with lock:
            update_prompt_controls(state['thread'] is not None and state['thread'].is_alive())

    @mode.on_update
    def change_mode(_):
        with lock:
            update_prompt_controls(state['thread'] is not None and state['thread'].is_alive())

    @preset_selection.on_update
    def load_prompt(_):
        with lock:
            if (state['thread'] is not None and state['thread'].is_alive()) or mode.value == 'Manual SAM prompts':
                return
            name = preset_selection.value
            if name != DEFAULT_NAME:
                loaded_prompt['loading'] = True
                try:
                    target.value = saved_prompts[name].get('target', 'non_traversable')
                finally:
                    loaded_prompt['loading'] = False
            instruction = instruction_for(target.value) if name == DEFAULT_NAME else saved_prompts[name]['instruction']
            loaded_prompt.update(name=name, instruction=instruction)
            prompt_editor.value = instruction
            preset_name.value = '' if name == DEFAULT_NAME else name
            prompt_notice.content = f'**Loaded:** {name}. Used on the next Start. Save under a new name to keep a variation.'

    @target.on_update
    def change_target(_):
        with lock:
            if loaded_prompt['loading'] or (state['thread'] is not None and state['thread'].is_alive()): return
            instruction = instruction_for(target.value)
            loaded_prompt.update(name=DEFAULT_NAME, instruction=instruction)
            prompt_editor.value = instruction
            preset_selection.value = DEFAULT_NAME

    @save_prompt.on_click
    def store_prompt(_):
        with lock:
            if (state['thread'] is not None and state['thread'].is_alive()) or mode.value == 'Manual SAM prompts':
                return
            try:
                value = presets.save(preset_name.value, prompt_editor.value, target.value)
            except (OSError, ValueError) as exc:
                prompt_notice.content = '**Cannot save:** '+str(exc)
                return
            saved_prompts[value['name']] = value
            preset_selection.options = [DEFAULT_NAME, *sorted(saved_prompts)]
            preset_selection.value = value['name']
            preset_name.value = value['name']
            loaded_prompt.update(name=value['name'], instruction=value['instruction'])
            prompt_notice.content = f'**Saved:** {value["name"]}. Saving the same name replaces that preset; previous runs keep their exact instruction.'

    @start.on_click
    def begin(_):
        with lock:
            if state['thread'] is not None and state['thread'].is_alive():
                return
            manual = mode.value == 'Manual SAM prompts'
            try:
                instruction = validate_instruction(prompt_editor.value) if mode.value == 'VLM proposals' else None
                manual_prompts = parse_sam_prompts(sam_editor.value) if manual else None
                extra_prompts = parse_sam_prompts(extra_editor.value, 64) if extra_enabled.value else []
                if not sam_enabled.value and (manual or extra_prompts or support_enabled.value):
                    raise ValueError('SAM queries require SAM segmentation enabled.')
                if planner_enabled.value and not costs_enabled.value:
                    raise ValueError('Enable Cost map before requesting Path planning.')
                if (slopes_enabled.value or costs_enabled.value or planner_enabled.value):
                    if not sam_enabled.value:
                        raise ValueError('Analysis requires SAM support/traversability evidence.')
                    if reference_source.value == 'base_support' and not support_enabled.value:
                        raise ValueError('Enable floor/ground support or explicitly select traversability reference evidence.')
                    if reference_source.value == 'traversability' and not (extra_prompts or (mode.value == 'VLM proposals' and target.value == 'traversable') or support_enabled.value):
                        raise ValueError('Traversability reference requires positive SAM/VLM queries.')
                selected_count = int(frame_count.value) if sampling.value == 'Offline frame count' else 0
                if selected_count and not 2 <= selected_count <= 1024:
                    raise ValueError('Choose between 2 and 1024 frames for offline processing.')
            except ValueError as exc:
                state['launch_error'] = str(exc)
                prompt_notice.content = '**Cannot start:** '+str(exc)
                return
            missing_vlm = mode.value == 'VLM proposals' and session.loaded and 'vlm' not in session.loaded_roles
            if not session.loaded or missing_vlm:
                status.content = overview_html({'status': 'checking GPU memory'})
                probe = subprocess.run([args.vlm_python, '-c',
                    'import json,torch; print(json.dumps({"free_gib":torch.cuda.mem_get_info()[0]/2**30}))'],
                    env={**os.environ,'PYTHONNOUSERSITE':'1'}, capture_output=True, text=True, timeout=30)
                if probe.returncode:
                    state['launch_error'] = 'GPU check failed; see the launcher log.'
                    print(probe.stderr, file=sys.stderr, flush=True)
                    return
                free = json.loads(probe.stdout)['free_gib']
                needed = 10. if missing_vlm else (11. if manual else 18.5)
                if free < needed:
                    state['launch_error'] = (f'GPU has {free:.2f} GiB free. This mode needs at least {needed:.1f} GiB free to start. '
                        'Free the GPU, then press Start processing again.')
                    return
            directory = args.output_root / f'{Path(selection.value).stem}_{uuid4().hex[:12]}'
            parameters = stream_args(command(args, videos[selection.value], directory,
                frame_count=selected_count, map_step=1 if map_every.value == 'Selected frame' else 2)[3:])
            parameters.vlm_instruction = instruction
            parameters.vlm_prompt_name = ('Manual SAM prompts' if manual else
                (loaded_prompt['name'] if instruction == loaded_prompt['instruction'] else 'Unsaved draft'))
            parameters.manual_sam_prompts = manual_prompts
            parameters.vlm_target = target.value
            parameters.no_vlm = mode.value != 'VLM proposals'
            parameters.no_sam = not sam_enabled.value
            parameters.extra_traversability_prompts = extra_prompts
            parameters.support_prompt = ('floor' if selection.value.startswith('indoor/') else 'ground') if support_enabled.value else None
            parameters.steepness, parameters.cost_map, parameters.plan = slopes_enabled.value, costs_enabled.value, planner_enabled.value
            parameters.analysis_workers = int(analysis_workers.value)
            parameters.reference_source = parameters.calibration_source = reference_source.value
            cancel.clear()
            def process_video():
                with directory.with_suffix('.log').open('w') as log:
                    try:
                        run(parameters, model_session=session, stop_event=cancel, output_stream=log)
                    except Exception as exc:
                        import traceback
                        traceback.print_exc(file=log)
                        with lock:
                            state['launch_error'] = str(exc)
            thread = threading.Thread(target=process_video, daemon=True)
            state.update(thread=thread, directory=directory, stop_requested=False, launch_error=None)
            start.disabled = selection.disabled = True
            update_prompt_controls(running=True)
            unload.disabled = True
            stop.disabled = False
            status.content = overview_html({'status': 'starting'}, roles=session.loaded_roles)
            thread.start()
            print(f'Live run started: {directory}', flush=True)

    def stop_owned_run():
        with lock:
            thread = state['thread']
            if thread is None or not thread.is_alive():
                return
            state['stop_requested'] = True
            cancel.set()

    @stop.on_click
    def halt(_):
        stop.disabled = True
        stop_owned_run()

    @unload.on_click
    def free_gpu(_):
        with lock:
            if state['thread'] is not None and state['thread'].is_alive():
                return
            session.close()

    finalized = {'directory': None, 'view': None, 'route': None}

    @colour_mode.on_update
    def change_colour(_):
        if finalized['view'] is not None and cloud is not None:
            cloud.colors = finalized['view']['modes'][colour_mode.value]

    @route_visible.on_update
    def change_route_visibility(_):
        if finalized['route'] is not None:
            finalized['route'].visible = route_visible.value

    current_directory, cloud, display_mtime, preview_source = None, None, None, None
    extent = 1.
    try:
        while True:
            with lock:
                thread, directory, stopped = state['thread'], state['directory'], state['stop_requested']
                launch_error = state['launch_error']
            if directory is None:
                status.content = overview_html({}, roles=session.loaded_roles,
                    message='Choose a video. Edit the Prompts tab if needed, then Start processing.', error=launch_error)
                time.sleep(.2)
                continue
            if directory != current_directory:
                if cloud is not None:
                    cloud.remove()
                pose_overlay.clear()
                follower.reset()
                current_directory = directory
                cloud, display_mtime, preview_source = None, None, None
                preview.visible = False
                if finalized['route'] is not None: finalized['route'].remove()
                finalized.update(directory=None, view=None, route=None)
                colour_mode.options = ['Traversability']
                colour_mode.value = 'Traversability'
            data = {}
            path = directory/'status.json'
            if path.exists():
                data = json.loads(path.read_text())
            message = ''
            if thread is None:
                message = 'Saved preview. Press Start processing for a new run.'
            elif not thread.is_alive():
                start.disabled = selection.disabled = False
                update_prompt_controls()
                unload.disabled = False
                stop.disabled = True
                if stopped:
                    message = 'Stopped. Partial output is saved. Press Start for a fresh run.'
                elif launch_error:
                    message = f'Processing failed. See {directory.name}.log in the output folder.'
                else:
                    message = 'Finished. The map remains visible. Press Start to run again.'
            status.content = overview_html(data,
                directory.relative_to(ROOT) if directory.is_relative_to(ROOT) else directory,
                session.loaded_roles, message, launch_error)
            display = directory/'display.npz'
            if display.exists() and display.stat().st_mtime_ns != display_mtime:
                mtime = display.stat().st_mtime_ns
                with np.load(display,allow_pickle=False) as prediction:
                    points, colors = prediction['points'].copy(), prediction['colors'].copy()
                if len(points):
                    lo,hi = np.percentile(points,[2,98],axis=0)
                    extent = max(float(np.max(hi-lo)),1e-3)
                    cloud = server.scene.add_point_cloud('/live_map',points=points,colors=colors,
                        point_size=extent*.001,precision='float32',point_shape='circle')
                display_mtime=mtime
            view_key = (directory, tuple(sorted(data.get('outputs', {}).items())))
            if data.get('status') in ('analysing', 'complete') and data.get('outputs', {}).get('map') and finalized['directory'] != view_key:
                from visualizer.integrated import load_view
                finished = load_view(directory)
                finalized.update(directory=view_key, view=finished)
                colour_mode.options = list(finished['modes'])
                colour_mode.value = 'Cost map' if 'Cost map' in finished['modes'] else 'Traversability'
                cloud = server.scene.add_point_cloud('/live_map', points=finished['points'],
                    colors=finished['modes'][colour_mode.value], point_size=extent*.001,
                    precision='float32', point_shape='circle')
                if len(finished['route']) > 1:
                    route_points = finished['route']
                    finalized['route'] = server.scene.add_line_segments('/verified_route',
                        points=np.stack((route_points[:-1], route_points[1:]), axis=1).astype(np.float32),
                        colors=(240, 0, 220), line_width=3.)
                    finalized['route'].visible = route_visible.value
                legend.content = ('Green: positive support evidence. Red: non-traversable veto. '
                    'Steepness: 0–90°. Route: verified against this snapshot.\n\n'
                    + ('Path planning: ' + finished['report']['status'] + '\n\n' if finished['report'] else '')
                    + '\n\n'.join(finished['notes']))
            pose = data.get('camera_pose')
            if pose:
                pose_overlay.update(pose,extent)
                pose_overlay.show_camera(True)
                follower.update(pose, extent, follow=follow_camera.value,
                    distance=follow_distance.value, height=follow_height.value, fov_degrees=follow_fov.value)
            source=data.get('source_index')
            if source is not None and source != preview_source:
                name=f'frame_{source:06d}.npz'
                prediction,mask_path=directory/'predictions'/name,directory/'masks'/name
                if prediction.exists() and mask_path.exists():
                    rgb=read_rgb(prediction)
                    with np.load(mask_path,allow_pickle=False) as saved:
                        mask=saved['out_binary_masks']
                    preview.content = previews_html(rgb, mask, data.get('source_time', 0.))
                    preview.visible = True
                    preview_source=source
            time.sleep(.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_owned_run()
        thread = state['thread']
        if thread is not None:
            thread.join()
        session.close()
        server.stop()


if __name__=='__main__':
    main()
