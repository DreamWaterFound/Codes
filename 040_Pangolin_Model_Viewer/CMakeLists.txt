# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)

cmake_minimum_required(VERSION 2.8)

project(test)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


find_package(Pangolin 0.3 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS} "/home/guoqing/SLAM/ThirdLib/Pangolin/include")
include_directories(${CMAKE_SOURCE_DIR}/inc)



add_executable(ModelViewer ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(ModelViewer ${Pangolin_LIBRARIES})
