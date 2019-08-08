cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(Pangolin_demo)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


# Eigen
find_package(Eigen 3 REQUIRED)
message("-- ==> Found Eigen ${EIGEN3_VERSION} in ${EIGEN_INCLUDE_DIR}")
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR}/..)

# Pangolin
find_package(Pangolin REQUIRED)
message("-- ==> Found Pangolin ${Pangolin_VERSION} in ${Pangolin_INCLUDE_DIRS}")
include_directories(${Pangolin_INCLUDE_DIRS})


include_directories(${PROJECT_SOURCE_DIR}/inc)

add_executable(demo ${PROJECT_SOURCE_DIR}/src/demo.cc)
target_link_libraries(demo ${Pangolin_LIBRARIES})

