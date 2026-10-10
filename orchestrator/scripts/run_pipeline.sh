#!/usr/bin/env bash
set -euo pipefail
usage() { echo 'Usage: run_pipeline.sh VIDEO PROMPT NEW_RUN_DIR [--view]'; }
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ $# -lt 3 || $# -gt 4 ]]; then usage >&2; exit 2; fi
if [[ $# -eq 4 && "$4" != --view ]]; then usage >&2; exit 2; fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
video="$(realpath -m -- "$1")"
prompt="$2"
run_dir="$(realpath -m -- "$3")"
require_file "$video"
check_prompt "$prompt"
new_output_dir "$run_dir"
mkdir -- "$run_dir/logs"
video_name="$(basename -- "$video")"
video_name="${video_name%.*}"

printf 'Run directory: %s\n' "$run_dir"
printf 'Stage 1/3: LingBot predictions (video rendering disabled)\n'
bash "$SCRIPTS_DIR/run_lingbot.sh" "$video" "$run_dir/lingbot" \
    2>&1 | tee "$run_dir/logs/lingbot.log"
printf 'Stage 2/3: SAM3 masks\n'
bash "$SCRIPTS_DIR/run_sam3.sh" "$video" "$prompt" "$run_dir/sam3" \
    2>&1 | tee "$run_dir/logs/sam3.log"
printf 'Stage 3/3: Semantic point cloud\n'
bash "$SCRIPTS_DIR/build_map.sh" "$video" "$prompt" \
    "$run_dir/lingbot/$video_name" "$run_dir/sam3/${video_name}_masks" \
    "$run_dir/semantic_map.ply" 2>&1 | tee "$run_dir/logs/map.log"
printf 'Pipeline completed. PLY: %s\n' "$run_dir/semantic_map.ply"

if [[ "${4:-}" == --view ]]; then
    bash "$SCRIPTS_DIR/view_map.sh" "$run_dir/semantic_map.ply" \
        2>&1 | tee "$run_dir/logs/viewer.log"
fi
