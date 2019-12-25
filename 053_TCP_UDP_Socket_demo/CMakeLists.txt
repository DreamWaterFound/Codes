cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(TCP_Scoket_Test)

# 编译器标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g")
set(CMAKE_BUILD_TYPE Release)

# 可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(TCP_Client ${PROJECT_SOURCE_DIR}/src/TCP_Client.cc)
add_executable(TCP_Server ${PROJECT_SOURCE_DIR}/src/TCP_Server.cc)

add_executable(UDP_Client ${PROJECT_SOURCE_DIR}/src/UDP_Client.cc)
add_executable(UDP_Server ${PROJECT_SOURCE_DIR}/src/UDP_Server.cc)

