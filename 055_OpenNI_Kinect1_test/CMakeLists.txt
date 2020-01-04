cmake_minimum_required(VERSION 2.8)

project(openni_kinect_test)

# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")
set(CMAKE_BUILD_TYPE Release)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 寻找库
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "==> OpenCV_INCLUDE_DIRS = " ${OpenCV_INCLUDE_DIRS})

include_directories("/usr/include/openni2")
include_directories("/usr/include/ni")

include_directories("/home/guoqing/libraries/eigen_333/include/eigen3")

# 生成文件
add_executable(openni2_test ${CMAKE_SOURCE_DIR}/src/main_openni2.cc)
target_link_libraries(openni2_test ${OpenCV_LIBS}  "/usr/lib/libOpenNI.so" "/home/guoqing/software/OpenNI2_other/baidwangpan/OpenNI-Linux-x64-2.2/Tools/libOpenNI2.so")

# failed
add_executable(openni1_test ${CMAKE_SOURCE_DIR}/src/main_openni1.cc)
target_link_libraries(openni1_test ${OpenCV_LIBS} "/usr/lib/libOpenNI.so")


