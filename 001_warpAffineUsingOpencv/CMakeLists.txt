cmake_minimum_required(VERSION 2.8)

project(warpAffine)

# 包含opencv库就足够了
find_package(OpenCV 3 REQUIRED)

if(OPENCV_FOUND)
    message(STATUS "OpenCV Found, include path: " ${OpenCV_INCLUDE_DIRS})
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(warpAffine main.cpp)
    target_link_libraries(warpAffine ${OpenCV_LIBS})
else(OPENCV_FOUND)
    message(FATAL "CAN NOT found OpenCV !")
endif(OPENCV_FOUND)

