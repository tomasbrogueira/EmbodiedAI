#!/usr/bin/env bash
set -euo pipefail
PLUGIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PLUGIN_ROOT/../.."
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/jovyan/.local/share/lingbot-runtime/viewer:/home/jovyan/.local/share/lingbot-runtime/models${PYTHONPATH:+:$PYTHONPATH}"
exec /opt/conda/bin/python plugins/small_obstacles/live_demo.py "$@"
