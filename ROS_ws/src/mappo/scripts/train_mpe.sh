#!/usr/bin/env bash
# Legacy MPE launcher, retargeted to the DOMAS training entry point.
# Usage: bash src/mappo/scripts/train_mpe.sh [--check]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PACKAGE_ROOT

# ---- Python runtime preflight (see tools/python_preflight.sh) --------------
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

ENV_NAME="DOMAS"
SCENARIO="simple_spread"
NUM_AGENTS="${NUM_AGENTS:-4}"
SEED_MAX="${SEED_MAX:-1}"
CONFIG_PATH="$(domas_native_path "${SCRIPT_DIR}/../config/train_config.yaml")"

for seed in $(seq 1 "${SEED_MAX}"); do
    echo "seed is ${seed}:"
    "${DOMAS_PYTHON}" -m mappo.train \
        --env "${ENV_NAME}" \
        --scenario_name "${SCENARIO}" \
        --num_agents "${NUM_AGENTS}" \
        --seed "${seed}" \
        --n_rollout_threads 1 \
        --episode_length 25 \
        --config_path "${CONFIG_PATH}"
done
