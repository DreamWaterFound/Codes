cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11")

project(background_subtraction)

find_package(OpenCV 3 REQUIRED)

if(OPENCV_FOUND)

    message(STATUS "OpenCV Found!")
    message(STATUS "Include directory:" ${OpenCV_INCLUDE_DIRS})

    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(motionDetect_backSub main.cpp)

    target_link_libraries(motionDetect_backSub ${OpenCV_LIBS})

else(OPENCV_FOUND)
message(ERROR "OpenCV NOT Found!")
endif(OPENCV_FOUND)


