cmake_minimum_required(VERSION 3.16)
project(panda_ik CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(MUJOCO_DIR $ENV{HOME}/libs/mujoco-3.16.0)
find_package(Eigen3 REQUIRED)

add_executable(ik ik.cpp)
target_include_directories(ik PRIVATE ${MUJOCO_DIR}/include)
target_link_libraries(ik PRIVATE ${MUJOCO_DIR}/lib/libmujoco.so Eigen3::Eigen)
set_target_properties(ik PROPERTIES BUILD_RPATH ${MUJOCO_DIR}/lib)
target_compile_definitions(ik PRIVATE
  MODEL_PATH="${CMAKE_SOURCE_DIR}/../mujoco_menagerie/franka_emika_panda/scene.xml")
