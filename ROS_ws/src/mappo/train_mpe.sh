#!/usr/bin/env bash
# Launch MAPPO training for the DOMAS four-TurtleBot environment.
# Usage: bash src/mappo/train_mpe.sh [--check] [extra train.py args...]
#
# --check runs only the Python runtime preflight: it verifies that the
# interpreter can import `encodings` and that training dependencies exist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PACKAGE_ROOT

# ---- Python runtime preflight ---------------------------------------------
# "Fatal Python error: Failed to import encodings module" is an INTERPRETER
# startup failure (invalid PYTHONHOME / broken venv prefix), never a source bug.
# Recover here and report it clearly instead of dying with a cryptic crash.
# Git Bash reports POSIX paths (/d/...) but a Win32 Python needs D:\...
# when PYTHONPATH is a ';'-separated list.  On Linux/cygpath-less hosts the
# path is already correct and is passed through unchanged.
domas_native_path() {
    if [ "${DOMAS_PATHSEP:-}" = ";" ] && command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$1"
    else
        printf '%s' "$1"
    fi
}

domas_bootstrap() {
    if [ -n "${PYTHONHOME:-}" ]; then
        echo "[domas] WARNING: PYTHONHOME='${PYTHONHOME}' overrides the interpreter prefix." >&2
        echo "[domas]          Unsetting it for this run (common cause of 'No module named encodings')." >&2
        unset PYTHONHOME
    fi

    local c
    DOMAS_PYTHON=""
    for c in "${DOMAS_PYTHON:-}" python3 python; do
        [ -n "$c" ] || continue
        command -v "$c" >/dev/null 2>&1 || continue
        if "$c" -c 'import encodings; assert encodings.__file__' >/dev/null 2>&1; then
            DOMAS_PYTHON="$c"
            break
        fi
        echo "[domas] interpreter '$c' failed the encodings probe; skipping." >&2
    done

    if [ -z "$DOMAS_PYTHON" ]; then
        echo "[domas] FATAL: no healthy Python interpreter (cannot import 'encodings')." >&2
        echo "[domas] Fix: clear PYTHONHOME in the launching shell, or deactivate a broken venv." >&2
        echo "[domas] Run 'bash tools/python_preflight.sh' for a full diagnosis." >&2
        return 1
    fi
    export DOMAS_PYTHON

    # Win32 Python splits PYTHONPATH on ';', POSIX Python on ':'.
    DOMAS_PATHSEP="$("$DOMAS_PYTHON" -c 'import os; print(os.pathsep)')"
    export DOMAS_PATHSEP
    local native_root
    native_root="$(domas_native_path "$PACKAGE_ROOT")"
    if [ -n "${PYTHONPATH:-}" ]; then
        export PYTHONPATH="${native_root}${DOMAS_PATHSEP}${PYTHONPATH}"
    else
        export PYTHONPATH="$native_root"
    fi
}

domas_report_deps() {
    local missing
    missing="$("$DOMAS_PYTHON" -c 'import importlib.util as u; print(" ".join(m for m in ("numpy","torch") if u.find_spec(m) is None))')" || {
        echo "[domas] dependency probe failed" >&2
        return 1
    }
    if [ -n "$missing" ]; then
        echo "[domas] Missing training dependencies: ${missing}" >&2
        echo "[domas] Install with: \"$DOMAS_PYTHON\" -m pip install numpy torch" >&2
        return 1
    fi
    return 0
}

if [ "${1:-}" = "--check" ]; then
    domas_bootstrap || exit 1
    "$DOMAS_PYTHON" -c 'import sys, encodings; print("encodings :", encodings.__file__); print("prefix    :", sys.prefix); print("python    :", sys.version.split()[0])'
    "$DOMAS_PYTHON" -c 'import mappo; print("mappo     :", mappo.__file__)' || {
        echo "[domas] FATAL: interpreter is healthy but cannot import 'mappo'." >&2
        echo "[domas] PYTHONPATH was: ${PYTHONPATH}" >&2
        exit 1
    }
    domas_report_deps || echo "[domas] preflight itself is OK; install the packages above before training."
    echo "[domas] preflight OK -- interpreter starts and imports encodings + mappo."
    exit 0
fi

domas_bootstrap || exit 1
domas_report_deps || exit 1
# ---------------------------------------------------------------------------

NUM_AGENTS="${NUM_AGENTS:-4}"
OBS_DIM="${OBS_DIM:-28}"
ACT_DIM="${ACT_DIM:-2}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config/train_config.yaml}"
CONFIG_PATH="$(domas_native_path "$CONFIG_PATH")"

"${DOMAS_PYTHON}" -m mappo.train \
    --env "DOMAS" \
    --num_agents "${NUM_AGENTS}" \
    --obs_dim "${OBS_DIM}" \
    --act_dim "${ACT_DIM}" \
    --config_path "${CONFIG_PATH}" \
    --n_rollout_threads 1 \
    --episode_length 25 "$@"
