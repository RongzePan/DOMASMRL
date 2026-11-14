#!/usr/bin/env python
"""catkin setuptools entry point for the consensus package.

The Python package lives at the catkin package root (not in src/<pkg>), so the
explicit ``package_dir`` mapping is required for ``catkin_python_setup()``.
"""
from setuptools import find_packages, setup
from catkin_pkg.python_setup import generate_distutils_setup

_root_packages = find_packages(where=".", exclude=("config", "config.*"))
packages = ["consensus"] + ["consensus." + p for p in _root_packages]

setup_args = generate_distutils_setup(
    packages=packages,
    package_dir={"consensus": "."},
    package_data={"consensus": ["config/*.yaml"]},
    version="0.1.0",
    install_requires=[],
    zip_safe=False,
)

setup(**setup_args)
