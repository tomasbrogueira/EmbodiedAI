#!/usr/bin/env bash
set -euo pipefail
usage() { echo 'Usage: view_map.sh PLY [VIEWER_OPTIONS...]'; }
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ $# -lt 1 ]]; then usage >&2; exit 2; fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
ply="$(realpath -m -- "$1")"
shift
require_file "$ply"
activate_env orchestrator-visualizer
exec python3 -u orchestrator/view_ply.py "$ply" "$@"
