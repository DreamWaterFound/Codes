cmake_minimum_required(VERSION 2.8)

project(Gauss_Filter)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_INCLUDE_DIRS})
list(APPEND LINK_LIBS ${OpenCV_LIBS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})
