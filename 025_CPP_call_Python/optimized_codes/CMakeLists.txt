cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(optimized_codes_for_yolact)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 这里使用的是我自己编译的Python版本

set(PYTHON_LIBRARY "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/my_py_pt/lib/libpython3.7m.a")
set(PYTHON_INCLUDE_DIR "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/my_py_pt/include/python3.7m")

add_subdirectory(src)


