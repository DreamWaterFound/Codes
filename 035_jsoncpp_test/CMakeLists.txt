cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(jsoncpp_test)

# find_package(jsoncpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(JSONCPP_INCLUDE_DIR "/home/guoqing/libraries/jsoncpp_190/include")
set(JSONCPP_LIBRARIES_DIR "/home/guoqing/libraries/jsoncpp_190/include")
set(JSONCPP_LIBS "libjsoncpp.a")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${JSONCPP_INCLUDE_DIR})
link_directories(${JSONCPP_LIBRARIES_DIR})

add_subdirectory(src)





