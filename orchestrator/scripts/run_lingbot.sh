#!/usr/bin/env bash
set -euo pipefail
usage() { echo 'Usage: run_lingbot.sh VIDEO NEW_OUTPUT_DIR'; }
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ $# -ne 2 ]]; then usage >&2; exit 2; fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
video="$(realpath -m -- "$1")"
output="$(realpath -m -- "$2")"
require_file "$video"
require_file "$PROJECT_ROOT/lingbot-map/lingbot-map.pt"
require_file "$PROJECT_ROOT/lingbot-map/demo_render/batch_demo.py"
require_new_path "$output"
activate_env lingbot-map
new_output_dir "$output"
# --no_render is the approved exception: save geometry without CUDA video rendering.
exec python3 -u lingbot-map/demo_render/batch_demo.py \
    --video_path "$video" \
    --output_folder "$output" \
    --model_path lingbot-map/lingbot-map.pt \
    --mode windowed --window_size 32 --keyframe_interval 2 \
    --num_scale_frames 2 --overlap_keyframes 8 --use_sdpa \
    --config demo_render/config/indoor.yaml \
    --save_predictions --no_render
