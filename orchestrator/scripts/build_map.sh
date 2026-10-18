#!/usr/bin/env bash
set -euo pipefail
usage() { echo 'Usage: build_map.sh VIDEO PROMPT LINGBOT_DIR SAM_MASK_DIR NEW_OUTPUT_PLY'; }
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ $# -ne 5 ]]; then usage >&2; exit 2; fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
video="$(realpath -m -- "$1")"
prompt="$2"
lingbot_dir="$(realpath -m -- "$3")"
sam_dir="$(realpath -m -- "$4")"
output="$(realpath -m -- "$5")"
require_file "$video"
require_dir "$lingbot_dir"
require_dir "$sam_dir"
check_prompt "$prompt"
require_new_path "$output"
activate_env orchestrator-visualizer
mkdir -p -- "$(dirname -- "$output")"
exec python -u orchestrator/main.py \
    --video-path "$video" --lingbot-dir "$lingbot_dir" --sam-dir "$sam_dir" \
    --output "$output" --prompt="$prompt" \
    --frame-step 2 --pixel-step 1 --min-depth-conf 2.5 --max-depth 25 \
    --patch-size 14 --segment-color 255,0,0
