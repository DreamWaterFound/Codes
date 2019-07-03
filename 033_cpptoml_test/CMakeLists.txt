cmake_minimum_required(VERSION 3.1.0)

project(cpptoml_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(src)