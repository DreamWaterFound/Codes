cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(NanoGUI_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

find_package(Threads REQUIRED)

# 需要 Eigen
include_directories("/home/guoqing/libraries/eigen_333/include/eigen3")

# NanoVG
include_directories("/home/guoqing/libraries/nanovg/src")

# NanoGUI
include_directories("/home/guoqing/libraries/nanogui_e9ec8a1/include")
link_directories("/home/guoqing/libraries/nanogui_e9ec8a1/lib")

add_executable(demo ${CMAKE_SOURCE_DIR}/src/demo.cc)
target_link_libraries(demo nanogui ${CMAKE_THREAD_LIBS_INIT})