cmake_minimum_required(VERSION 2.8)

project(GFlag_Demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(gflags_DIR "/home/guoqing/libraries/gflags_222/lib/cmake/gflags")

find_package(gflags)

include_directories(${GFLAGS_INCLUDE_DIR})

add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/main.cc)

target_link_libraries(${PROJECT_NAME} ${GFLAGS_LIBRARIES})
