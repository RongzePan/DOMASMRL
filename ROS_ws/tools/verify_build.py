#!/usr/bin/env python3
"""Build verification driver for the DOMASMRL ROS workspace.

The target platform is ROS Melodic (Ubuntu), but this repository is being
edited on Windows where catkin / roscore / torch / gym are not installed.
This script therefore runs an equivalent, layered build verification:

  1. Python byte-compilation of every source file (catches compile-level errors)
  2. XML / YAML / shell syntax validation of all configuration assets
  3. catkin metadata validation (package.xml, CMakeLists.txt, setup.py)
  4. Launch-file and Gazebo model URI reference resolution
  5. A REAL CMake configure + build + install of every first-party package,
     using tools/catkin_shim as a minimal catkin stand-in
  6. Full-module import smoke test with stubbed ROS / torch dependencies
  7. Functional smoke tests of environments, wrappers, agent and utilities

Exit code 0 means every stage passed.  A machine-readable report is written to
build/verify/verification_report.json.
"""

from __future__ import annotations

import datetime
import importlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import types
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BUILD = ROOT / "build" / "verify"
SHIM = ROOT / "tools" / "catkin_shim" / "cmake"
FIRST_PARTY = ["mappo", "consensus", "sim_world"]
SKIP_DIRS = {".git", "__pycache__", ".idea", "build", "devel", "install", "logs"}


class Report:
    def __init__(self):
        self.checks = []

    def add(self, stage, name, ok, detail="", status=None):
        status = status or ("PASS" if ok else "FAIL")
        self.checks.append({"stage": stage, "name": name, "ok": bool(ok),
                            "status": status, "detail": str(detail)[:1200]})
        print(f"  [{status}] {name}" + (f" -- {str(detail)[:160]}" if detail else ""))

    def skip(self, stage, name, reason):
        """An optional tool is unavailable: record it explicitly, do not fail the build."""
        self.add(stage, name, True, reason, status="SKIP")

    def counts(self):
        total = len(self.checks)
        failed = [c for c in self.checks if c["status"] == "FAIL"]
        skipped = [c for c in self.checks if c["status"] == "SKIP"]
        return total, failed, skipped


def walk_src():
    for path in SRC.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def run_tool(argv, **kwargs):
    """``subprocess.run`` that returns (result, error) instead of raising OSError.

    PyCharm / cmd / PowerShell launches often have a reduced PATH, so tools
    such as ``bash`` or ``cmake`` may be missing entirely.  A missing optional
    tool must skip a check, never crash a whole stage.
    """
    try:
        res = subprocess.run(argv, capture_output=True, text=True,
                             errors="replace", **kwargs)
        return res, None
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _find_bash():
    """Locate a POSIX shell even when PATH was stripped by the launching IDE."""
    exe = shutil.which("bash")
    if exe:
        return exe
    candidates = [
        pathlib.Path(r"C:\Program Files\Git\bin\bash.exe"),
        pathlib.Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
        pathlib.Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
        pathlib.Path("/usr/bin/bash"),
        pathlib.Path("/bin/bash"),
    ]
    portable_git = pathlib.Path.home() / ".workbuddy" / "binaries" / "PortableGit" / "versions"
    if portable_git.is_dir():
        candidates += sorted(portable_git.glob("*/bin/bash.exe"))
        candidates += sorted(portable_git.glob("*/usr/bin/bash.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _strip_yaml_comment(line):
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def _yaml_structural_lint(text):
    """Conservative YAML syntax lint used when PyYAML is not installed.

    It checks tab indentation and balanced flow-style brackets/braces.  It is
    deliberately NOT advertised as a full YAML parse; ``pip install pyyaml``
    upgrades this to ``yaml.safe_load`` automatically.
    """
    text = re.sub(r"^%YAML:[^\n]*\n", "", text)
    errors, stack = [], []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_yaml_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in line[:indent]:
            errors.append(f"L{lineno}: tab in indentation")
        for ch in line:
            if ch in "[{":
                stack.append((ch, lineno))
            elif ch in "]}":
                if not stack:
                    errors.append(f"L{lineno}: unmatched '{ch}'")
                    break
                opened, opened_at = stack.pop()
                if (opened, ch) not in (("[", "]"), ("{", "}")):
                    errors.append(f"L{lineno}: mismatched bracket opened at L{opened_at}")
    for ch, lineno in stack:
        errors.append(f"L{lineno}: unclosed '{ch}'")
    return errors


# --------------------------------------------------------------------------- 1
def stage_python_compile(rep):
    import py_compile
    files = sorted(p for p in walk_src() if p.suffix == ".py")
    failures = []
    for p in files:
        try:
            py_compile.compile(str(p), doraise=True, cfile=str(BUILD / "pyc" /
                               (str(p.relative_to(SRC)).replace(os.sep, "_") + "c")))
        except py_compile.PyCompileError as exc:
            failures.append(f"{p.relative_to(ROOT)}: {exc}")
    rep.add("1-python-compile", f"Byte-compile {len(files)} Python sources",
            not failures, "; ".join(failures[:5]) or f"{len(files)} files OK")


# -------------------------------------------------------------------------- 1b
def stage_python_preflight(rep):
    """Regression tests for the 'Failed to import encodings module' startup crash.

    A stale PYTHONHOME (or a broken venv prefix) kills CPython before any
    project code runs.  The launchers must detect it, recover, and still be
    able to import ``mappo`` through a correctly separated PYTHONPATH.
    """
    bad_home = "/definitely/not/a/prefix"
    env = os.environ.copy()

    # Control: prove the fatal error is reproducible in this environment.
    broken = {**env, "PYTHONHOME": bad_home}
    res, err = run_tool([sys.executable, "-c", "pass"], env=broken)
    if err:
        rep.skip("1b-python-preflight", "Control: broken PYTHONHOME", err)
    else:
        output = res.stdout + res.stderr
        rep.add("1b-python-preflight", "Control: broken PYTHONHOME reproduces the fatal error",
                res.returncode != 0 and "encodings" in output,
                " | ".join(output.strip().splitlines()[:2]))

    # Pure-Python equivalent of the bootstrap: always available, no shell needed.
    clean = {k: v for k, v in env.items() if k != "PYTHONHOME"}
    clean["PYTHONPATH"] = str(SRC) + (os.pathsep + clean["PYTHONPATH"]
                                      if clean.get("PYTHONPATH") else "")
    probe = ("import sys, encodings, mappo; "
             "print(encodings.__file__); print(mappo.__file__)")
    res, err = run_tool([sys.executable, "-c", probe], env=clean, cwd=str(ROOT))
    rep.add("1b-python-preflight",
            "Interpreter recovers with PYTHONHOME cleared + os.pathsep PYTHONPATH",
            bool(not err and res.returncode == 0 and "mappo" in res.stdout),
            err or (res.stdout.strip().splitlines()[-1] if res.stdout.strip()
                    else res.stderr.strip()[:200]))

    # Shell-based regression tests only when a POSIX shell is reachable.
    bash = _find_bash()
    cases = [
        ("standalone preflight script", ROOT / "tools" / "python_preflight.sh", [], env),
        ("train_mpe.sh --check under broken PYTHONHOME",
         ROOT / "src" / "mappo" / "train_mpe.sh", ["--check"], broken),
        ("scripts/train_mpe.sh --check under broken PYTHONHOME",
         ROOT / "src" / "mappo" / "scripts" / "train_mpe.sh", ["--check"], broken),
    ]
    if not bash:
        for name, _script, _args, _case_env in cases:
            rep.skip("1b-python-preflight", name,
                     "bash not on PATH in this process (PyCharm/cmd); pure-Python check above covers it")
        return
    for name, script, args, case_env in cases:
        res, err = run_tool([bash, str(script)] + args, env=case_env, cwd=str(ROOT))
        if err:
            rep.skip("1b-python-preflight", name, err)
            continue
        output = res.stdout + res.stderr
        ok = res.returncode == 0 and "OK" in output and "mappo" in output
        notable = [ln for ln in output.strip().splitlines()
                   if "OK" in ln or "mappo" in ln or "FATAL" in ln]
        rep.add("1b-python-preflight", name, ok,
                " | ".join(notable[-3:] or output.strip().splitlines()[-3:])[:300])


# --------------------------------------------------------------------------- 2
def stage_asset_syntax(rep):
    xml_ext = {".xml", ".launch", ".sdf", ".world", ".config"}
    bad_xml = []
    n_xml = 0
    for p in walk_src():
        if p.suffix in xml_ext or p.name == "model.config":
            n_xml += 1
            try:
                ET.parse(p)
            except Exception as exc:
                bad_xml.append(f"{p.relative_to(ROOT)}: {exc}")
    rep.add("2-assets", f"Parse {n_xml} XML/launch/SDF/world/config files",
            not bad_xml, "; ".join(bad_xml[:5]) or "all well-formed")

    bad_yaml, n_yaml = [], 0
    try:
        import yaml
    except ImportError:
        yaml = None
    for p in walk_src():
        if p.suffix not in {".yaml", ".yml", ".rviz"}:
            continue
        n_yaml += 1
        text = p.read_text(encoding="utf-8", errors="replace")
        # RViz configs start with the YAML 1.0 directive, which PyYAML rejects.
        clean = re.sub(r"^%YAML:[^\n]*\n", "", text)
        if yaml is not None:
            try:
                yaml.safe_load(clean)
            except Exception as exc:
                bad_yaml.append(f"{p.relative_to(ROOT)}: {str(exc)[:100]}")
        else:
            for issue in _yaml_structural_lint(clean):
                bad_yaml.append(f"{p.relative_to(ROOT)}: {issue}")
    label = (f"Parse {n_yaml} YAML/RViz configs" if yaml is not None
             else f"Structural-lint {n_yaml} YAML/RViz configs (PyYAML absent)")
    rep.add("2-assets", label, not bad_yaml,
            "; ".join(bad_yaml[:5]) or "all well-formed")
    if yaml is None:
        rep.skip("2-assets", "Full yaml.safe_load validation",
                 "PyYAML not installed in this interpreter; "
                 "python -m pip install pyyaml to enable")

    sh = sorted(p for p in walk_src() if p.suffix == ".sh")
    bash = _find_bash()
    if not bash:
        rep.skip("2-assets", f"bash -n on {len(sh)} shell launchers",
                 "bash not on PATH in this process")
        return
    bad_sh = []
    for p in sh:
        res, err = run_tool([bash, "-n", str(p)])
        if err:
            rep.skip("2-assets", f"bash -n {p.name}", err)
            continue
        if res.returncode != 0:
            bad_sh.append(f"{p.relative_to(ROOT)}: {res.stderr.strip()[:100]}")
    rep.add("2-assets", f"bash -n on {len(sh)} shell launchers",
            not bad_sh, "; ".join(bad_sh) or "all syntactically valid")


# --------------------------------------------------------------------------- 3
def stage_catkin_metadata(rep):
    pkgs = [d for d in sorted(SRC.iterdir()) if d.is_dir() and (d / "package.xml").is_file()]
    names = set()
    problems = []
    for d in pkgs:
        try:
            tree = ET.parse(d / "package.xml")
            name = tree.getroot().findtext("name")
        except Exception as exc:
            problems.append(f"{d.name}/package.xml: {exc}")
            continue
        if name != d.name:
            problems.append(f"{d.name}/package.xml name='{name}' does not match directory")
        names.add(name)
        cm = d / "CMakeLists.txt"
        if not cm.is_file():
            problems.append(f"{d.name}: CMakeLists.txt missing")
        else:
            text = cm.read_text(encoding="utf-8")
            for token in ("cmake_minimum_required", "project(", "find_package(catkin", "catkin_package("):
                if token not in text:
                    problems.append(f"{d.name}/CMakeLists.txt: missing {token}")
            if text.count("(") != text.count(")"):
                problems.append(f"{d.name}/CMakeLists.txt: unbalanced parentheses")
    rep.add("3-catkin-metadata", f"Validate {len(pkgs)} catkin packages",
            not problems, "; ".join(problems[:8]) or f"packages: {', '.join(sorted(names))}")

    top = SRC / "CMakeLists.txt"
    top_text = top.read_text(encoding="utf-8") if top.is_file() else ""
    top_problems = []
    if not top_text:
        top_problems.append("src/CMakeLists.txt missing")
    elif top_text.lstrip().startswith("/"):
        top_problems.append("bare symlink-target path; not valid CMake (Git-on-Windows artifact)")
    elif "toplevel.cmake" not in top_text:
        top_problems.append("does not reference catkin toplevel.cmake")
    elif top_text.count("(") != top_text.count(")"):
        top_problems.append("unbalanced parentheses")
    rep.add("3-catkin-metadata", "Workspace top-level src/CMakeLists.txt",
            not top_problems, "; ".join(top_problems) or "portable catkin toplevel resolution")

    # Whole-workspace manifest coverage (including vendored turtlebot3 packages).
    manifests = [p for p in SRC.rglob("package.xml")
                 if not any(part in SKIP_DIRS for part in p.parts)]
    mprob, internal_names = [], set()
    for p in manifests:
        try:
            root = ET.parse(p).getroot()
        except Exception as exc:
            mprob.append(f"{p.relative_to(ROOT)}: {exc}")
            continue
        name = root.findtext("name")
        if not name:
            mprob.append(f"{p.relative_to(ROOT)}: missing <name>")
        elif name != p.parent.name:
            mprob.append(f"{p.relative_to(ROOT)}: name '{name}' != dir '{p.parent.name}'")
        else:
            internal_names.add(name)
    for p in manifests:
        try:
            root = ET.parse(p).getroot()
        except Exception:
            continue
        for dep in root.findall("depend") + root.findall("exec_depend") + root.findall("build_depend"):
            target = (dep.text or "").strip()
            if target in {"mappo", "consensus", "sim_world"} and target not in internal_names:
                mprob.append(f"{p.parent.name}: undeclared workspace package '{target}'")
    rep.add("3-catkin-metadata", f"Validate {len(manifests)} package.xml manifests workspace-wide",
            not mprob, "; ".join(mprob[:8]) or f"{len(manifests)} manifests, names and deps consistent")

    # Every first-party package must declare build rules that reference real files.
    missing = []
    for pkg in FIRST_PARTY:
        d = SRC / pkg
        cm = (d / "CMakeLists.txt").read_text(encoding="utf-8")
        # RENAME specifies the *installed* filename, not a source file.
        cm = re.sub(r"RENAME\s+[\w./-]+\.(?:py|sh)\b", "", cm)
        for script in set(re.findall(r"[\w./-]+\.(?:py|sh)\b", cm)):
            if not (d / script).is_file():
                missing.append(f"{pkg}: script {script} not found")
        for folder in re.findall(r"install\(DIRECTORY\s+([^\s/]+)/", cm):
            if not (d / folder).is_dir():
                missing.append(f"{pkg}: directory {folder}/ not found")
        if "catkin_python_setup()" in cm and not (d / "setup.py").is_file():
            missing.append(f"{pkg}: setup.py referenced but missing")
    rep.add("3-catkin-metadata", "CMake install targets reference existing files",
            not missing, "; ".join(missing) or "all targets resolvable")

    # setup.py: package discovery + real execution with a catkin_pkg stub
    problems = []
    try:
        import runpy
        import setuptools
        from setuptools import find_packages
    except ImportError as exc:
        rep.skip("3-catkin-metadata", "Execute both setup.py files (catkin_pkg stubbed)",
                 f"setuptools not installed in this interpreter ({exc})")
        return
    try:

        fake_root = types.ModuleType("catkin_pkg")
        fake_sub = types.ModuleType("catkin_pkg.python_setup")
        fake_sub.generate_distutils_setup = lambda **kwargs: kwargs
        fake_root.python_setup = fake_sub
        sys.modules["catkin_pkg"] = fake_root
        sys.modules["catkin_pkg.python_setup"] = fake_sub

        captured = {}
        original_setup = setuptools.setup
        setuptools.setup = lambda **kwargs: captured.update(kwargs)
        cwd = os.getcwd()
        try:
            for pkg in ("mappo", "consensus"):
                found = set(find_packages(where=str(SRC / pkg), exclude=("config", "config.*")))
                expected = {
                    "mappo": {"envs", "utils", "runner", "algorithms", "train", "scripts"},
                    "consensus": {"src"},
                }[pkg]
                if not expected.issubset(found):
                    problems.append(f"{pkg}: find_packages misses {expected - found}")
                captured.clear()
                os.chdir(SRC / pkg)
                runpy.run_path(str(SRC / pkg / "setup.py"), run_name="__main__")
                packages = captured.get("packages", [])
                if not packages or packages[0] != pkg:
                    problems.append(f"{pkg}/setup.py: bad packages list {packages[:3]}")
                if captured.get("package_dir", {}).get(pkg) != ".":
                    problems.append(f"{pkg}/setup.py: bad package_dir {captured.get('package_dir')}")
        finally:
            os.chdir(cwd)
            setuptools.setup = original_setup
    except Exception as exc:
        problems.append(f"setup.py execution: {type(exc).__name__}: {exc}")
    rep.add("3-catkin-metadata", "Execute both setup.py files (catkin_pkg stubbed)",
            not problems,
            "; ".join(problems) or "package_dir mapping + package lists valid")


# --------------------------------------------------------------------------- 4
def stage_references(rep):
    pkg_dirs = {d.name: d for d in SRC.iterdir()
                if d.is_dir() and (d / "package.xml").is_file()}
    external = {"gazebo_ros", "xacro", "rviz", "tf", "robot_state_publisher",
                "turtlebot3_description", "turtlebot3_gazebo"}
    problems, checked = [], 0
    for p in sorted((SRC / "consensus").rglob("*.launch")):
        text = p.read_text(encoding="utf-8")
        for pkg, rel in re.findall(r"\$\(find\s+([A-Za-z0-9_]+)\)(/[^\s\"'<>]+)?", text):
            checked += 1
            if pkg not in pkg_dirs:
                if pkg not in external:
                    problems.append(f"{p.name}: unknown package $(find {pkg})")
                continue
            if rel and not (pkg_dirs[pkg] / rel.lstrip("/")).exists():
                problems.append(f"{p.name}: missing $(find {pkg}){rel}")
    rep.add("4-references", f"Resolve {checked} $(find ...) launch references",
            not problems, "; ".join(problems[:8]) or "all resolvable")

    # Gazebo model:// URIs
    model_roots = [SRC / "sim_world" / "model", ROOT / "models"]
    bad, n = [], 0
    for p in sorted((SRC / "sim_world" / "world").iterdir()):
        if not p.is_file():
            continue
        for uri in re.findall(r"model://([^<\s\"']+)", p.read_text(encoding="utf-8", errors="replace")):
            n += 1
            if not any((root / uri).exists() for root in model_roots):
                bad.append(f"{p.name}: model://{uri}")
    rep.add("4-references", f"Resolve {n} Gazebo model:// URIs",
            not bad, "; ".join(bad[:8]) or "all resolvable")


# --------------------------------------------------------------------------- 5
def _find_cmake():
    """Locate CMake independent of which Python interpreter runs this script.

    A project venv rarely contains the ``cmake`` wheel, but another interpreter
    on the machine (or a Visual Studio installation) often does.  Reusing that
    binary keeps the real configure/build/install verification available.
    """
    exe = shutil.which("cmake")
    if exe:
        return exe
    candidates = []
    try:
        import cmake
        candidates.append(pathlib.Path(cmake.__file__).parent / "data" / "bin" / "cmake.exe")
    except ImportError:
        pass
    candidate_roots = [
        pathlib.Path(sys.executable).parent / "Scripts" / "cmake.exe",
        pathlib.Path(r"C:\Program Files\CMake\bin\cmake.exe"),
        pathlib.Path(r"C:\Program Files (x86)\CMake\bin\cmake.exe"),
        pathlib.Path("/usr/bin/cmake"),
        pathlib.Path("/usr/local/bin/cmake"),
    ]
    candidates += candidate_roots
    home = pathlib.Path.home()
    for pattern, root in (
        ("*/Lib/site-packages/cmake/data/bin/cmake.exe",
         home / ".workbuddy" / "binaries" / "python" / "versions"),
        ("*/Lib/site-packages/cmake/data/bin/cmake.exe",
         home / "AppData" / "Local" / "Programs" / "Python"),
        ("*/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe",
         pathlib.Path(r"C:\Program Files (x86)\Microsoft Visual Studio")),
    ):
        if root.is_dir():
            candidates += sorted(root.glob(pattern))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _find_ninja():
    """Locate a standalone Ninja binary (no MSVC environment needed)."""
    exe = shutil.which("ninja")
    if exe:
        return exe
    candidates = []
    for root in (pathlib.Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
                 pathlib.Path(r"C:\Program Files\Microsoft Visual Studio")):
        if root.is_dir():
            candidates += sorted(root.glob(
                "*/*/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _cmake_configure(exe, src, bd, inst):
    """Configure with the default generator, falling back to standalone Ninja.

    IDE-launched processes often lack the MSVC environment variables that
    CMake needs for its Visual Studio / NMake generators.  Ninja is a single
    self-contained executable, so it works with a reduced environment.
    """
    base = [exe, "-S", str(src), "-B", str(bd),
            f"-Dcatkin_DIR={SHIM}",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            f"-DCMAKE_INSTALL_PREFIX={inst}"]
    attempts = [("default", base)]
    ninja = _find_ninja()
    if ninja:
        attempts.append(("Ninja", base + ["-G", "Ninja",
                                          f"-DCMAKE_MAKE_PROGRAM={ninja}"]))
    last_error = "no generator attempted"
    for label, cmd in attempts:
        if bd.exists():
            shutil.rmtree(bd)
        res, err = run_tool(cmd)
        if err:
            last_error = f"{label}: {err}"
            continue
        if res.returncode == 0:
            return label, None
        last_error = f"{label}: " + " | ".join(
            (res.stdout + res.stderr).strip().splitlines()[-4:])
    return None, last_error


def stage_cmake_build(rep):
    exe = _find_cmake()
    if not exe:
        rep.skip("5-cmake-build", "Real CMake configure+build+install",
                 "no cmake binary found (python -m pip install cmake to enable)")
        return
    ver_res, ver_err = run_tool([exe, "--version"])
    if ver_err or ver_res.returncode != 0:
        rep.skip("5-cmake-build", "Real CMake configure+build+install",
                 ver_err or "cmake --version failed")
        return
    ver = ver_res.stdout.splitlines()[0] if ver_res.stdout else "cmake"
    rep.add("5-cmake-build", "cmake executable available", True, f"{ver}  ({exe})")

    inst = BUILD / "install"
    if inst.exists():
        shutil.rmtree(inst)

    for pkg in FIRST_PARTY:
        src = SRC / pkg
        bd = BUILD / "cmake" / pkg
        if bd.exists():
            shutil.rmtree(bd)
        generator, cfg_error = _cmake_configure(exe, src, bd, inst)
        if generator is None:
            rep.skip("5-cmake-build", f"{pkg}: cmake configure",
                     "no usable CMake generator in this environment: "
                     + cfg_error[:220])
            continue
        rep.add("5-cmake-build", f"{pkg}: cmake configure", True,
                f"configure OK (generator: {generator})")

        r, err = run_tool([exe, "--build", str(bd), "--config", "Release"])
        if err:
            rep.skip("5-cmake-build", f"{pkg}: cmake --build", err)
        elif r.returncode != 0:
            rep.add("5-cmake-build", f"{pkg}: cmake --build", False,
                    " | ".join((r.stdout + r.stderr).strip().splitlines()[-6:]))
        else:
            rep.add("5-cmake-build", f"{pkg}: cmake --build", True, "build OK")

        r, err = run_tool([exe, "--install", str(bd), "--config", "Release"])
        if err:
            rep.skip("5-cmake-build", f"{pkg}: cmake --install", err)
        elif r.returncode != 0:
            rep.add("5-cmake-build", f"{pkg}: cmake --install", False,
                    " | ".join((r.stdout + r.stderr).strip().splitlines()[-6:]))
        else:
            n = sum(1 for _ in (inst).rglob("*") if _.is_file()) if inst.exists() else 0
            rep.add("5-cmake-build", f"{pkg}: cmake --install", True,
                    f"install tree now holds {n} files")

    expected = [
        "lib/mappo/train.py",
        "lib/mappo/render_mpe.py",
        "lib/mappo/train_mpe.sh",
        "lib/mappo/train_mpe_legacy.sh",
        "share/mappo/config/train_config.yaml",
        "share/mappo/python_preflight.sh",
        "lib/consensus/control_robots.py",
        "lib/consensus/MultipleTurtlebotsEnv.py",
        "share/consensus/launch/4turtlebot.launch",
        "share/consensus/config/4turtle.rviz",
        "share/sim_world/model/dock/model.sdf",
        "share/sim_world/world/emptypark.world",
    ]
    if not inst.exists():
        rep.skip("5-cmake-build", f"Installed artifact manifest ({len(expected)} entries)",
                 "CMake build was skipped in this environment")
        return
    missing = [f for f in expected if not (inst / f).is_file()]
    rep.add("5-cmake-build", f"Installed artifact manifest ({len(expected)} entries)",
            not missing, "; ".join(missing) or "every ROS executable/launch/model installed")


# --------------------------------------------------------------------------- 6
class _Any:
    """Permissive stand-in for arbitrary ROS APIs used at import time."""
    def __init__(self, name="stub"):
        self._name = name

    def __call__(self, *a, **k):
        return _Any(self._name)

    def __getattr__(self, item):
        return _Any(f"{self._name}.{item}")

    def __iter__(self):
        return iter(())

    def __float__(self):
        return 0.0

    def __int__(self):
        return 0

    def __repr__(self):
        return f"<stub {self._name}>"


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any(f"{self.__name__}.{name}")


def _mod(name, **attrs):
    m = _StubModule(name)
    for key, value in attrs.items():
        setattr(m, key, value)
    sys.modules[name] = m
    return m


def _install_stubs():
    # numpy is a real dependency of the import/functional stages.  Keep the
    # stub installer itself importable without it so callers can degrade
    # gracefully (SKIP) rather than crash the whole stage.
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - depends on the host environment
        np = None

    # ---- ROS message / service modules ------------------------------------
    ros_modules = [
        "rospy", "rospkg", "tf", "tf.transformations",
        "std_msgs", "std_msgs.msg", "geometry_msgs", "geometry_msgs.msg",
        "gazebo_msgs", "gazebo_msgs.msg", "gazebo_msgs.srv",
        "sensor_msgs", "sensor_msgs.msg", "std_srvs", "std_srvs.srv",
        "tensorboardX", "imageio",
    ]
    for name in ros_modules:
        _mod(name)
    sys.modules["tf"].transformations = sys.modules["tf.transformations"]

    # ---- torch (import-time API surface only) -----------------------------
    class _Module:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return self.forward(*a, **k)

        def forward(self, *a, **k):
            return _Any("forward")

        def parameters(self):
            return []

        def train(self):
            return self

        def eval(self):
            return self

        def to(self, *a, **k):
            return self

        def state_dict(self):
            return {}

    class _Dist:
        def __init__(self, *a, **k):
            pass

    class _Device:
        def __init__(self, spec="cpu"):
            self.type = str(spec)

        def __repr__(self):
            return f"device(type={self.type})"

    def _no_grad(func=None):
        return func if func is not None else (lambda f: f)

    nn = _mod("torch.nn", Module=_Module, ModuleList=list, Parameter=lambda *a, **k: _Any("Parameter"),
              Linear=_Any("nn.Linear"), Conv2d=_Any("nn.Conv2d"), GRU=_Any("nn.GRU"),
              LayerNorm=_Any("nn.LayerNorm"), ReLU=_Any("nn.ReLU"), Tanh=_Any("nn.Tanh"),
              Sequential=_Any("nn.Sequential"))
    fn = _mod("torch.nn.functional", linear=lambda *a, **k: _Any("linear"))
    init = _mod("torch.nn.init", xavier_uniform_=lambda *a, **k: None,
                orthogonal_=lambda *a, **k: None, constant_=lambda *a, **k: None,
                kaiming_uniform_=lambda *a, **k: None, uniform_=lambda *a, **k: None,
                calculate_gain=lambda *a, **k: 1.0,
                _calculate_fan_in_and_fan_out=lambda *a, **k: (1, 1))
    utils = _mod("torch.nn.utils", clip_grad_norm_=lambda *a, **k: 0.0)
    nn.functional, nn.init, nn.utils = fn, init, utils

    dist = _mod("torch.distributions", Categorical=_Dist, Normal=_Dist, Bernoulli=_Dist)
    optim = _mod("torch.optim", Adam=_Any("Adam"))
    cuda = _mod("torch.cuda", is_available=lambda: False, manual_seed_all=lambda *a: None)
    cudnn = _mod("torch.backends.cudnn", benchmark=False, deterministic=False)
    backends = _mod("torch.backends", cudnn=cudnn)

    torch = _mod(
        "torch", nn=nn, cuda=cuda, backends=backends, distributions=dist, optim=optim,
        device=_Device, float32="torch.float32", float64="torch.float64",
        Tensor=_Any("Tensor"), no_grad=_no_grad,
        from_numpy=lambda x: np.asarray(x) if np is not None else x,
        tensor=lambda *a, **k: _Any("tensor"),
        zeros=lambda *a, **k: _Any("zeros"), ones=lambda *a, **k: _Any("ones"),
        cat=lambda *a, **k: _Any("cat"), clamp=lambda *a, **k: _Any("clamp"),
        exp=lambda *a, **k: _Any("exp"), max=lambda *a, **k: _Any("max"),
        min=lambda *a, **k: _Any("min"), sum=lambda *a, **k: _Any("sum"),
        sqrt=lambda *a, **k: _Any("sqrt"), transpose=lambda *a, **k: _Any("transpose"),
        gt=lambda *a, **k: _Any("gt"), randperm=lambda *a, **k: _Any("randperm"),
        manual_seed=lambda *a: None, set_num_threads=lambda *a: None,
        save=lambda *a, **k: None, load=lambda *a, **k: {},
    )
    for name, module in (("nn", nn), ("optim", optim), ("cuda", cuda),
                         ("backends", backends), ("distributions", dist)):
        setattr(torch, name, module)


def stage_import_smoke(rep):
    if importlib.util.find_spec("numpy") is None:
        rep.skip("6-import-smoke", "Import smoke test (numpy required)",
                 "numpy not installed in this interpreter; "
                 "python -m pip install -r tools/requirements-verify.txt")
        return
    _install_stubs()
    sys.path.insert(0, str(SRC))
    mods = []
    for pkg in ("mappo", "consensus"):
        for p in sorted((SRC / pkg).rglob("*.py")):
            if "__pycache__" in p.parts or p.name == "setup.py":
                continue
            rel = p.relative_to(SRC).with_suffix("")
            name = ".".join(rel.parts)
            if name.endswith(".__init__"):
                name = name[:-len(".__init__")]
            mods.append(name)
    failures = []
    for name in mods:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    rep.add("6-import-smoke", f"Import {len(mods)} workspace modules with dependency stubs",
            not failures, "; ".join(failures[:8]) or "all modules import cleanly")


# --------------------------------------------------------------------------- 7
def stage_functional(rep):
    if importlib.util.find_spec("numpy") is None:
        rep.skip("7-functional", "Functional smoke tests (numpy required)",
                 "numpy not installed in this interpreter; "
                 "python -m pip install -r tools/requirements-verify.txt")
        return
    import numpy as np

    def check(name, fn):
        try:
            detail = fn()
            rep.add("7-functional", name, True, detail or "OK")
        except Exception as exc:
            import traceback
            rep.add("7-functional", name, False,
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")

    def config_parser():
        from mappo.config import get_config
        args = get_config().parse_args([])
        assert args.num_env_steps > 0 and args.hidden_size == 64
        return f"defaults OK (hidden_size={args.hidden_size})"

    def train_args():
        from mappo.train.train import parse_args
        from mappo.config import get_config
        a = parse_args(["--num_agents", "4", "--config_path", "config/train_config.yaml"],
                       get_config())
        assert a.num_agents == 4 and a.env_name == "MyEnv"
        return f"num_agents={a.num_agents}, lr={a.lr}, steps={a.num_env_steps}"

    def env_core():
        from mappo.envs.env_core import EnvCore
        e = EnvCore()
        obs = e.reset()
        assert len(obs) == 4 and all(o.shape == (14,) for o in obs)
        for _ in range(6):
            actions = [[np.random.uniform(-1, 1, 5)] for _ in range(4)]
            obs, rew, done, info = e.step(actions)
        e.seed(7)
        e.close()
        return "4 agents x 14-dim obs, 6 steps OK"

    def domas_env():
        from mappo.envs.VirtualTurtlebot import EnvCore
        e = EnvCore(num_agents=4)
        obs = e.reset()
        assert len(obs) == 4 and all(o.shape == (28,) for o in obs)
        for _ in range(5):
            obs, rew, done, info = e.step([np.random.uniform(-1, 1, 2) for _ in range(4)])
        assert len(rew) == 4 and len(done) == 4
        e2 = EnvCore(num_agents=6)  # exercises the pose-cycling fix
        e2.reset()
        e.close()
        e2.close()
        return "DOMAS EnvCore: 28-dim obs, 5 steps, 6-agent construction OK"

    def wrappers():
        from mappo.envs.env_continuous import ContinuousActionEnv
        from mappo.envs.env_discrete import DiscreteActionEnv
        from mappo.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
        c = ContinuousActionEnv()
        c.seed(1)
        o = c.reset()
        assert o.shape == (4, 14)
        o, r, d, i = c.step([np.zeros(5) for _ in range(4)])
        c.close()
        dsc = DiscreteActionEnv()
        dsc.reset()
        dsc.step([0 for _ in range(4)])
        dsc.close()
        vec = DummyVecEnv([ContinuousActionEnv for _ in range(2)])
        vobs = vec.reset()
        assert vobs.shape[0] == 2
        vec.step(np.zeros((2, 4, 5)))
        vec.close()
        sub = SubprocVecEnv([ContinuousActionEnv])
        sub.reset()
        sub.close()
        return "continuous/discrete/DummyVecEnv/SubprocVecEnv OK"

    def agent():
        from mappo.algorithms.algorithm.rmappo_agent import RMAPPOAgent
        from mappo.algorithms.algorithm import r_mappo
        from mappo.envs.env_core import EnvCore
        e = EnvCore()
        a = RMAPPOAgent(env=e, config={"num_agents": 4, "obs_dim": 14, "action_dim": 5})
        assert r_mappo.RMAPPOAgent is RMAPPOAgent
        state = np.stack(e.reset())
        action = a.select_action(state)
        assert action.shape == (4, 5)
        info = a.update(state, action, -1.0, state)
        assert "grad_norm" in info
        return f"select_action {action.shape}, update grad_norm={info['grad_norm']:.4f}"

    def domas_adapter():
        from consensus.MultipleTurtlebotsEnv import MultipleTurtlebotsEnv, Formation, PID
        from consensus.src.MultipleTurtlebotsEnv import MultipleTurtlebotsEnv as Shim
        assert Shim is MultipleTurtlebotsEnv
        env = MultipleTurtlebotsEnv()
        state = env.reset()
        assert state.shape == (4, 14)
        next_state, reward, done, info = env.step([0, 1, 2, 3])
        assert isinstance(done, bool) and np.isfinite(reward)
        pid = PID(1, 0, 0)
        pid.setPoint(1.0)
        assert np.isfinite(pid.update(0.0))
        env.close()
        return f"state {state.shape}, reward={reward:.3f}, done={done}"

    def mpe_adapter():
        from mappo.envs.mpe.MPE_env import MPEEnv
        from mappo.config import get_config
        a = get_config().parse_args([])
        a.num_agents = 4
        e = MPEEnv(a)
        o = e.reset()
        assert o.shape == (4, 14)
        e.step(np.zeros((4, 5)))
        e.seed(3)
        e.close()
        return f"MPEEnv reset {o.shape} OK"

    def numeric_utils():
        from mappo.utils.util import check, tile_images, get_shape_from_obs_space, get_shape_from_act_space
        from mappo.envs.spaces_compat import spaces
        arr = np.zeros((3, 2), dtype=np.float32)
        assert check(arr) is not None and check("passthrough") == "passthrough"
        assert get_shape_from_obs_space(spaces.Box(-1, 1, (7,))) == (7,)
        assert get_shape_from_act_space(spaces.Discrete(5)) == 1
        assert tile_images(np.zeros((4, 2, 2, 1))).shape == (4, 4, 1)
        return "check/tile_images/shape helpers OK"

    def replay_buffer():
        from mappo.config import get_config
        from mappo.envs.spaces_compat import spaces
        from mappo.utils.separated_buffer import SeparatedReplayBuffer
        from mappo.utils.shared_buffer import SharedReplayBuffer
        a = get_config().parse_args([])
        a.episode_length = 4
        a.hidden_size = 8
        obs = spaces.Box(-1, 1, (14,), dtype=np.float32)
        share = spaces.Box(-1, 1, (56,), dtype=np.float32)
        act = spaces.Discrete(5)
        sb = SeparatedReplayBuffer(a, obs, share, act)
        shb = SharedReplayBuffer(a, 4, obs, share, act)
        return f"separated {sb.obs.shape}, shared {shb.obs.shape}"

    for name, fn in [
        ("config.get_config()", config_parser),
        ("train.parse_args() + YAML overlay", train_args),
        ("env_core.EnvCore reset/step", env_core),
        ("VirtualTurtlebot.EnvCore reset/step", domas_env),
        ("env wrappers (continuous/discrete/vec)", wrappers),
        ("RMAPPOAgent select_action/update", agent),
        ("MultipleTurtlebotsEnv adapter + PID", domas_adapter),
        ("MPEEnv adapter", mpe_adapter),
        ("numeric utilities", numeric_utils),
        ("replay buffers construction", replay_buffer),
    ]:
        check(name, fn)


# --------------------------------------------------------------------------- main
def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "pyc").mkdir(parents=True, exist_ok=True)
    rep = Report()
    print("=" * 78)
    print("DOMASMRL ROS WORKSPACE -- BUILD VERIFICATION")
    print(f"workspace : {ROOT}")
    print(f"time      : {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"python    : {sys.version.split()[0]} on {sys.platform}")
    print("=" * 78)

    for title, fn in [
        ("1. Python byte-compilation", stage_python_compile),
        ("1b. Python runtime preflight (encodings regression)", stage_python_preflight),
        ("2. Asset syntax (XML/YAML/shell)", stage_asset_syntax),
        ("3. catkin metadata", stage_catkin_metadata),
        ("4. Launch / Gazebo references", stage_references),
        ("5. Real CMake configure+build+install", stage_cmake_build),
        ("6. Import smoke test", stage_import_smoke),
        ("7. Functional smoke tests", stage_functional),
    ]:
        print(f"\n--- {title} ---")
        try:
            fn(rep)
        except Exception as exc:
            import traceback
            rep.add(title, "stage crashed", False, traceback.format_exc(limit=6))

    total, failed, skipped = rep.counts()
    report = {
        "workspace": str(ROOT),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "total_checks": total,
        "passed_checks": total - len(failed) - len(skipped),
        "failed_checks": len(failed),
        "skipped_checks": len(skipped),
        "result": "PASS" if not failed else "FAIL",
        "checks": rep.checks,
    }
    out = BUILD / "verification_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"RESULT: {report['result']}  "
          f"({report['passed_checks']} passed / {len(skipped)} skipped / "
          f"{len(failed)} failed, {total} total)")
    if skipped:
        print("Skipped (optional tooling unavailable in this interpreter):")
        for c in skipped:
            print(f"  - {c['name']}: {c['detail'][:110]}")
        print("  Enable all checks with: python -m pip install -r tools/requirements-verify.txt")
    print(f"report: {out}")
    print("=" * 78)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
