cmake_minimum_required(VERSION 2.8)

project(Panglion_Study)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


# Eigen
find_package(Eigen 3 REQUIRED)
message("-- ==> Found Eigen ${EIGEN3_VERSION} in ${EIGEN_INCLUDE_DIR}")
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR}/..)


find_package(Pangolin REQUIRED)
message("-- ==> Found Pangolin ${Pangolin_VERSION} in ${Pangolin_INCLUDE_DIRS}")
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(00_project_test src/00_project_test.cpp)

add_executable(01_HelloPangolin src/01_HelloPangolin.cpp)
target_link_libraries(01_HelloPangolin ${Pangolin_LIBRARIES})

add_executable(02_HelloPangolinOffscreen src/02_HelloPangolinOffscreen.cpp)
target_link_libraries(02_HelloPangolinOffscreen ${Pangolin_LIBRARIES})

add_executable(03_HelloPangolinThreads src/03_HelloPangolinThreads.cpp)
target_link_libraries(03_HelloPangolinThreads ${Pangolin_LIBRARIES})

add_executable(04_SharedMemoryCamera src/04_SharedMemoryCamera.cpp)
target_link_libraries(04_SharedMemoryCamera ${Pangolin_LIBRARIES})

add_executable(05_SimpleDisplay src/05_SimpleDisplay.cpp)
target_link_libraries(05_SimpleDisplay ${Pangolin_LIBRARIES})