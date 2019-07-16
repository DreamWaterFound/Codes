cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(jsoncpp_test)

# find_package(jsoncpp)

set(JSONCPP_INCLUDE_DIR "/home/guoqing/libraries/jsoncpp_190/include")
set(JSONCPP_LIBRARIES_DIR "/home/guoqing/libraries/jsoncpp_190/include")
set(JSONCPP_LIBS "libjsoncpp.a")


include_directories(${JSONCPP_INCLUDE_DIR})
link_directories(${JSONCPP_LIBRARIES_DIR})





