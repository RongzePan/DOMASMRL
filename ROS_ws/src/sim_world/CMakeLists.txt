cmake_minimum_required(VERSION 3.0.2)
# Pure Python/data package: no C/C++ targets, so do not require a
# compiler.  This keeps `catkin_make`/CMake working on hosts whose
# build toolchain is not on PATH.
project(sim_world LANGUAGES NONE)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY model/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/model
)

install(DIRECTORY world/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/world
)
