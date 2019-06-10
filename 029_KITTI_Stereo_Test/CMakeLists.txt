cmake_minimum_required(VERSION 2.8)

project(KITTI_Stereo_Test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_moudles)

# Eigen3
find_package(Eigen 3 REQUIRED)
message("-- ==> Found Eigen ${EIGEN3_VERSION} in ${EIGEN_INCLUDE_DIR}")
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR}/..)

# Pangolin
find_package(Pangolin REQUIRED)
message("-- ==> Found Pangolin ${Pangolin_VERSION} in ${Pangolin_INCLUDE_DIRS}")
include_directories(${Pangolin_INCLUDE_DIRS})
list(APPEND LINK_LIBS ${Pangolin_LIBRARIES})

# OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("-- ==> Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")
list(APPEND LINK_LIBS ${OpenCV_LIBS})

# 接下来也可以尝试着寻找 pangolin? PCL? 但是目前先不管那么多,先把视差图得到再说



add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})





