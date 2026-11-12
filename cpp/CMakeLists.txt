cmake_minimum_required(VERSION 3.16)
project(panda_ik CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(MUJOCO_DIR $ENV{HOME}/libs/mujoco-3.16.0)
set(MODEL_PATH "${CMAKE_SOURCE_DIR}/../mujoco_menagerie/franka_emika_panda/scene.xml")
find_package(Eigen3 REQUIRED)

# Reusable solver (PIC so it can go into a Python module later)
add_library(ik_core STATIC ik.cpp)
set_target_properties(ik_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ik_core PUBLIC ${CMAKE_SOURCE_DIR} ${MUJOCO_DIR}/include)
target_link_libraries(ik_core PUBLIC ${MUJOCO_DIR}/lib/libmujoco.so Eigen3::Eigen)

# Test + benchmark program
add_executable(ik main.cpp)
target_link_libraries(ik PRIVATE ik_core)
set_target_properties(ik PROPERTIES BUILD_RPATH ${MUJOCO_DIR}/lib)
target_compile_definitions(ik PRIVATE MODEL_PATH="${MODEL_PATH}")

# Python module (import panda_ik)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(panda_ik bindings.cpp)
target_link_libraries(panda_ik PRIVATE ik_core)
set_target_properties(panda_ik PROPERTIES
  BUILD_RPATH ${MUJOCO_DIR}/lib
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/..)
