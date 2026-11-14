#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Python environment preflight for the DOMAS launchers.
#
# Diagnoses / repairs the startup failure:
#     Could not find platform independent libraries <prefix>
#     Fatal Python error: Failed to import encodings module
#     ModuleNotFoundError: No module named 'encodings'
#
# That error happens BEFORE any Python code runs: the interpreter's prefix
# (PYTHONHOME, or a broken pyvenv.cfg / ._pth) does not contain Lib/encodings.
# It is never caused by project source code.
#
# Usage:
#   bash tools/python_preflight.sh                 # standalone diagnosis
#   source tools/python_preflight.sh
#   domas_preflight <package_root>                # repair + export DOMAS_PYTHON
#
# Windows note: the Win32 Python interpreter splits PYTHONPATH on ';' while a
# POSIX one splits on ':'.  We ask the chosen interpreter for os.pathsep.
# ---------------------------------------------------------------------------

domas_diag() { printf '[domas-preflight] %s\n' "$*" >&2; }

domas_interpreter_healthy() {
    "$1" -c 'import encodings, sys; assert encodings.__file__' >/dev/null 2>&1
}

# Git Bash reports /d/... ; a Win32 Python needs D:\... once PYTHONPATH is a
# ';'-separated list (MSYS no longer auto-converts list elements).
domas_native_path() {
    if [ "${DOMAS_PATHSEP:-}" = ";" ] && command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$1"
    else
        printf '%s' "$1"
    fi
}

domas_preflight() {
    local package_root="${1:-}"
    local candidate sep

    # 1. A stale PYTHONHOME is the dominant cause of the fatal error.
    if [ -n "${PYTHONHOME:-}" ]; then
        domas_diag "PYTHONHOME='${PYTHONHOME}' overrides the interpreter prefix."
        domas_diag "Unsetting it for this run (it must point at a full CPython prefix or be absent)."
        unset PYTHONHOME
    fi

    # 2. Select an interpreter that can actually import the standard library.
    DOMAS_PYTHON=""
    for candidate in "${DOMAS_PYTHON:-}" python3 python; do
        [ -n "$candidate" ] || continue
        command -v "$candidate" >/dev/null 2>&1 || continue
        if domas_interpreter_healthy "$candidate"; then
            DOMAS_PYTHON="$candidate"
            break
        fi
        domas_diag "interpreter '$candidate' failed the encodings probe; skipping."
    done

    if [ -z "${DOMAS_PYTHON:-}" ]; then
        cat >&2 <<'EOF'
[domas-preflight] FATAL: no healthy Python interpreter found.
  Fix the environment in the shell that launches the script:
    Git Bash / Linux / macOS : unset PYTHONHOME
    PowerShell               : Remove-Item Env:PYTHONHOME
    cmd.exe                  : set PYTHONHOME=
  Verify with: python -c "import encodings; print(encodings.__file__)"
  If a venv is active, deactivate it or recreate it (its base interpreter is gone).
EOF
        return 1
    fi
    export DOMAS_PYTHON

    # 3. Build PYTHONPATH using the interpreter's own separator.
    DOMAS_PATHSEP="$("$DOMAS_PYTHON" -c 'import os; print(os.pathsep)')"
    export DOMAS_PATHSEP
    local sep="$DOMAS_PATHSEP" native_root
    native_root="$(domas_native_path "$package_root")"
    if [ -n "$package_root" ]; then
        if [ -n "${PYTHONPATH:-}" ]; then
            export PYTHONPATH="${native_root}${sep}${PYTHONPATH}"
        else
            export PYTHONPATH="$native_root"
        fi
    fi
    domas_diag "interpreter : $(command -v "$DOMAS_PYTHON")"
    domas_diag "version     : $("$DOMAS_PYTHON" --version 2>&1)"
    domas_diag "path sep    : '${sep}'"
    [ -z "$package_root" ] || domas_diag "package root: ${native_root}"
    return 0
}

# Standalone mode: print a diagnosis and exit with a usable status.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" 2>/dev/null && pwd)" || root=""
    domas_preflight "$root" || exit 1
    "$DOMAS_PYTHON" -c "import sys, encodings; print('encodings:', encodings.__file__); print('prefix   :', sys.prefix)"
    "$DOMAS_PYTHON" -c "import mappo; print('mappo    :', mappo.__file__)" || {
        domas_diag "FATAL: interpreter works but 'mappo' is not importable."
        domas_diag "PYTHONPATH was: ${PYTHONPATH:-<empty>}"
        exit 1
    }
    echo "[domas-preflight] OK"
fi
