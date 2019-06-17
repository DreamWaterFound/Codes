cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

project(test_pytorch_env)

# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_moudles)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


find_package(Threads)


# set(Python_ADDITIONAL_VERSIONS "3.7")
set(PYTHON_LIBRARY "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/tmp_pt/lib/libpython3.7m.a")
set(PYTHON_INCLUDE_DIR "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/tmp_pt/include/python3.7m")

find_package(PythonLibs REQUIRED)


if(PYTHONLIBS_FOUND)
    message(STATUS "Python Lib Found! Are: " ${PYTHON_LIBRARIES})
    include_directories(${PYTHON_INCLUDE_DIRS})
    add_executable(${PROJECT_NAME} ./main.cpp)
    target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} dl util)
else(PYTHONLIBS_FOUND)
    message(STATUS "Python Lib NOT FOUND!")
endif(PYTHONLIBS_FOUND)
