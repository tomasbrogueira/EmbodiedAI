#!/usr/bin/env bash
set -euo pipefail
usage() { echo 'Usage: run_sam3.sh VIDEO PROMPT NEW_OUTPUT_DIR'; }
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ $# -ne 3 ]]; then usage >&2; exit 2; fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
video="$(realpath -m -- "$1")"
prompt="$2"
output="$(realpath -m -- "$3")"
require_file "$video"
require_file "$PROJECT_ROOT/sam3/generate_sam3_masks.py"
check_prompt "$prompt"
require_new_path "$output"
activate_env sam3
new_output_dir "$output"
exec python3 -u sam3/generate_sam3_masks.py \
    --video_path "$video" --output_folder "$output" --prompt="$prompt"
