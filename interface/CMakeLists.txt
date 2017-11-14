cmake_minimum_required(VERSION 2.8.7)
project(interface)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  cv_bridge
  low_level_vision
  pose_estimation
)

find_package(OpenCV 2.4.8 REQUIRED)
find_package(Eigen REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(CUDA REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -lhdf5")

set(LIBDEP
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${HDF5_LIBRARIES}
  ${CUDA_LIBRARIES}
)
message(${HDF5_LIBRARIES})

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME} ${LIBDEP}
  CATKIN_DEPENDS low_level_vision pose_estimation
  DEPENDS cv_bridge HDF5
)

# show additional files in qtcreator
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f
  OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME} SOURCES
  ${FILES_LIST}
  ${PROJECT_SOURCE_DIR}/../README.md
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV2_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
  ${HDF5_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/multi_rigid_tracker.cpp
  src/multi_rigid_detector.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${LIBDEP} -lhdf5
)

add_executable(cmd_line_generate_sift_model
  src/cmd_line_generate_sift_model.cpp
)
target_link_libraries(cmd_line_generate_sift_model
  ${PROJECT_NAME}
#  ${LIBDEP}
)

add_executable(cmd_line_generate_sift_model_separate
  src/cmd_line_generate_sift_model_separate.cpp
)
target_link_libraries(cmd_line_generate_sift_model_separate
  ${PROJECT_NAME}
#  ${LIBDEP}
)

add_executable(cmd_line_optical_flow
  src/cmd_line_optical_flow.cpp
)
target_link_libraries(cmd_line_optical_flow
  ${PROJECT_NAME}
#  ${LIBDEP}
)

add_executable(cmd_line_render
  src/cmd_line_render.cpp
)
target_link_libraries(cmd_line_render
  ${PROJECT_NAME}
#  ${LIBDEP}
)

add_executable(cmd_line_render_scene
  src/cmd_line_render_scene.cpp
)
target_link_libraries(cmd_line_render_scene
  ${PROJECT_NAME}
#  ${LIBDEP}
)

add_executable(cmd_line_get_robot_frame
  src/cmd_line_get_robot_frame.cpp
)
target_link_libraries(cmd_line_get_robot_frame
  ${PROJECT_NAME}
#  ${LIBDEP}
)

add_executable(cmd_line_extract_sift
  src/cmd_line_extract_sift.cpp
)
target_link_libraries(cmd_line_extract_sift
  ${PROJECT_NAME}
#  ${LIBDEP}
)

add_executable(cmd_line_detect_object_pose
  src/cmd_line_detect_object_pose.cpp
)
target_link_libraries(cmd_line_detect_object_pose
  ${PROJECT_NAME}
#  ${LIBDEP}
)

add_executable(cmd_line_track_multi_object
  src/cmd_line_track_multi_object.cpp
)
target_link_libraries(cmd_line_track_multi_object
  ${PROJECT_NAME}
#  ${LIBDEP}
)
