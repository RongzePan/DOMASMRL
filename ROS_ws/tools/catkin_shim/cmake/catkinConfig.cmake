# ---------------------------------------------------------------------------
# Minimal catkinConfig.cmake shim for OFFLINE BUILD VERIFICATION.
#
# This workspace targets ROS Melodic (Ubuntu + /opt/ros/melodic).  On a Windows
# host catkin is unavailable, so this file implements only the catkin CMake API
# used by the three first-party, pure-Python packages (mappo, consensus,
# sim_world).  It lets a real `cmake` configure/build/install run verify:
#   * package.xml / setup.py presence,
#   * every catkin_install_python() script actually exists,
#   * every install(DIRECTORY ...) source actually exists,
#   * the install tree is produced correctly.
#
# It is NOT a replacement for catkin on the target robot: a real catkin build
# still uses /opt/ros/melodic/share/catkin/cmake/toplevel.cmake.
# ---------------------------------------------------------------------------

set(catkin_FOUND TRUE)
set(catkin_VERSION "0.8.10")
set(catkin_INCLUDE_DIRS "")
set(catkin_LIBRARY_DIRS "")
set(catkin_LIBRARIES "")
set(catkin_EXPORTED_TARGETS "")
set(catkin_DEVEL_PREFIX "${CMAKE_BINARY_DIR}/devel")

# catkin destination variables (normally exported by catkin_package()).
set(CATKIN_PACKAGE_BIN_DESTINATION "lib/${PROJECT_NAME}")
set(CATKIN_PACKAGE_SHARE_DESTINATION "share/${PROJECT_NAME}")
set(CATKIN_PACKAGE_INCLUDE_DESTINATION "include/${PROJECT_NAME}")
set(CATKIN_PACKAGE_LIB_DESTINATION "lib")
set(CATKIN_GLOBAL_BIN_DESTINATION "bin")

function(catkin_package)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/package.xml")
    message(FATAL_ERROR "catkin_package(): package.xml is missing in ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  message(STATUS "[catkin-shim] catkin_package(${PROJECT_NAME}) -- metadata OK")
endfunction()

function(catkin_python_setup)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
    message(FATAL_ERROR "catkin_python_setup(): setup.py is missing in ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  message(STATUS "[catkin-shim] catkin_python_setup() -- setup.py OK")
endfunction()

function(catkin_install_python)
  cmake_parse_arguments(ARG "" "DESTINATION" "PROGRAMS" ${ARGN})
  if(NOT ARG_PROGRAMS)
    message(FATAL_ERROR "catkin_install_python(): no PROGRAMS given")
  endif()
  if(NOT ARG_DESTINATION)
    message(FATAL_ERROR "catkin_install_python(): no DESTINATION given")
  endif()
  foreach(_script IN LISTS ARG_PROGRAMS)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_script}")
      message(FATAL_ERROR "catkin_install_python(): script not found: ${_script}")
    endif()
  endforeach()
  install(PROGRAMS ${ARG_PROGRAMS} DESTINATION "${ARG_DESTINATION}")
  message(STATUS "[catkin-shim] installing python programs: ${ARG_PROGRAMS}")
endfunction()

# No-op message/action generators (used only by third-party packages).
function(add_message_files)
endfunction()
function(add_service_files)
endfunction()
function(add_action_files)
endfunction()
function(generate_messages)
endfunction()
function(generate_dynamic_reconfigure_options)
endfunction()
function(catkin_add_gtest)
endfunction()
function(catkin_add_nosetests)
endfunction()
