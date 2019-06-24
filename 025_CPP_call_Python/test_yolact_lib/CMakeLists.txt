cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(test_yolact_cpp_lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)



set(PYTHON_LIBRARY "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/tmp_pt/lib/libpython3.7m.a")
set(PYTHON_INCLUDE_DIR "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/tmp_pt/include/python3.7m")
set(CMAKE_LIBRARY_PATH ${PROJECT_SOURCE_DIR}/../test_image_transform/lib/)

find_package(Threads REQUIRED)


find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})


# OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("-- ==> Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")

include_directories(${PROJECT_SOURCE_DIR}/../test_image_transform/include)
# link_libraries("../test_image_transform/lib")



add_executable(app_test main.cpp)
target_link_libraries(app_test  ${OpenCV_LIBS} )
target_link_libraries(app_test /home/guoqing/Codes/025_CPP_call_Python/test_image_transform/lib/libyolact_cpp_lib.a)
target_link_libraries(app_test ${PYTHON_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS} dl util)


