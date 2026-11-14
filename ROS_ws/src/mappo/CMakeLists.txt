cmake_minimum_required(VERSION 3.0.2)
# Pure Python/data package: no C/C++ targets, so do not require a
# compiler.  This keeps `catkin_make`/CMake working on hosts whose
# build toolchain is not on PATH.
project(mappo LANGUAGES NONE)

find_package(catkin REQUIRED)

## Install the `mappo` Python package (see setup.py) into the devel space.
catkin_python_setup()

catkin_package()

## Python entry points. catkin_install_python rewrites the shebang to the
## interpreter selected at configure time.
catkin_install_python(PROGRAMS
  train/train.py
  scripts/render/render_mpe.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Shell launchers are not Python scripts: install them verbatim.
## Both files are named train_mpe.sh, so the legacy one is renamed to avoid
## one silently overwriting the other in lib/mappo/.
install(PROGRAMS train_mpe.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(PROGRAMS scripts/train_mpe.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  RENAME train_mpe_legacy.sh
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

## Standalone interpreter diagnostic for the target robot.
install(FILES ../../tools/python_preflight.sh
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
