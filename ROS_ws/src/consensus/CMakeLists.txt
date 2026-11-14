cmake_minimum_required(VERSION 3.0.2)
# Pure Python/data package: no C/C++ targets, so do not require a
# compiler.  This keeps `catkin_make`/CMake working on hosts whose
# build toolchain is not on PATH.
project(consensus LANGUAGES NONE)

find_package(catkin REQUIRED)

## Install the `consensus` Python package (see setup.py).
catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  src/control_robots.py
  MultipleTurtlebotsEnv.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
