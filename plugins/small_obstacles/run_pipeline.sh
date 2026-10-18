#!/usr/bin/env bash
# Run from anywhere; each stage can use a different existing environment.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 LINGBOT_PREDICTIONS_DIR OUTPUT_DIR" >&2
    exit 2
fi

predictions_dir=$(realpath "$1")
plugin_output_dir=$(realpath -m "$2")
plugin_repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$plugin_repo_dir"

"${VLM_PYTHON:-python}" -m plugins.small_obstacles propose \
    --lingbot-dir "$predictions_dir" \
    --output "$plugin_output_dir/prompts.json" \
    --model "${VLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}" \
    --keyframe-step "${KEYFRAME_STEP:-30}"

"${SAM_PYTHON:-python}" -m plugins.small_obstacles segment \
    --lingbot-dir "$predictions_dir" \
    --prompts "$plugin_output_dir/prompts.json" \
    --output-dir "$plugin_output_dir/masks" \
    --frame-step "${FRAME_STEP:-2}"

"${PROJECT_PYTHON:-python}" -m plugins.small_obstacles project \
    --lingbot-dir "$predictions_dir" \
    --masks-dir "$plugin_output_dir/masks" \
    --output "$plugin_output_dir/obstacle_map.ply"
