cmake_minimum_required(VERSION 2.8 )

project(core_dump_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
# 增加 gcc 的符号输出
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(test_core_dump src/main.cc)

