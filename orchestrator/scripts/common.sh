#!/usr/bin/env bash
# Shared setup for the scripts in this directory; source, do not execute.
SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPTS_DIR/../.." && pwd)"
# Relative arguments consistently refer to the repository root.
cd -- "$PROJECT_ROOT"

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || fail "File not found: $1"; }
require_dir() { [[ -d "$1" ]] || fail "Directory not found: $1"; }
require_new_path() {
    [[ ! -e "$1" && ! -L "$1" ]] || fail "Output already exists: $1. Choose a new path."
}
new_output_dir() {
    require_new_path "$1"
    mkdir -p -- "$(dirname -- "$1")"
    mkdir -- "$1"
}
check_prompt() {
    [[ -n "${1//[[:space:]]/}" ]] || fail 'The prompt must not be empty.'
    # SAM3 places the prompt verbatim in each mask filename.
    case "$1" in
        *'/'*|*'\'*|*$'\n'*|*$'\r'*) fail 'Prompt cannot contain slashes or newlines.' ;;
    esac
}
activate_env() {
    local prefix="$PROJECT_ROOT/.setup/envs/$1"
    local conda_sh="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
    require_file "$prefix/bin/python"
    require_file "$conda_sh"
    # Conda activation hooks may reference variables that are not set yet.
    set +u
    source "$conda_sh"
    conda activate "$prefix"
    set -u
    [[ "$CONDA_PREFIX" == "$prefix" ]] || fail "Wrong environment: $CONDA_PREFIX"
    export HF_HOME="$PROJECT_ROOT/.setup/cache/huggingface"
    export MPLCONFIGDIR="$PROJECT_ROOT/.setup/cache/matplotlib"
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
}
