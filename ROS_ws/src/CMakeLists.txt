# Catkin workspace top-level CMakeLists.txt.
#
# In a native ROS Melodic install this file is a SYMLINK to
#   /opt/ros/melodic/share/catkin/cmake/toplevel.cmake
# Git on Windows cannot store that symlink, so it was checked out as a plain
# text file containing the link target -- which makes `catkin_make` fail with
# "Unknown CMake command /opt/ros/melodic/...".  Resolve the real file at
# configure time instead, which works both on Linux and in checkouts made on
# Windows.
cmake_minimum_required(VERSION 3.0.2)

set(CATKIN_TOPLEVEL_CMAKE "/opt/ros/melodic/share/catkin/cmake/toplevel.cmake")

if(EXISTS "${CATKIN_TOPLEVEL_CMAKE}")
  include("${CATKIN_TOPLEVEL_CMAKE}")
else()
  message(FATAL_ERROR
    "catkin toplevel.cmake not found at ${CATKIN_TOPLEVEL_CMAKE}. "
    "Source your ROS installation first: source /opt/ros/melodic/setup.bash")
endif()
