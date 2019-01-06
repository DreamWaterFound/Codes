cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11")

project(erode_and_dilate)

find_package(OpenCV 3 REQUIRED)

if(OPENCV_FOUND)

    message(STATUS "OpenCV Found!")
    message(STATUS "Include directory:" ${OpenCV_INCLUDE_DIRS})

    include_directories(${OpenCV_INCLUDE_DIRS})

    add_executable(${PROJECT_NAME} main.cpp)

    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

else(OPENCV_FOUND)
message(ERROR "OpenCV NOT Found!")
endif(OPENCV_FOUND)