cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(image_transform_test)

# 注意，现在要求使用g++ 4.8来进行编译

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(PYTHON_LIBRARY "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/tmp_pt/lib/libpython3.7m.a")
set(PYTHON_INCLUDE_DIR "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/tmp_pt/include/python3.7m")

add_subdirectory(src)


