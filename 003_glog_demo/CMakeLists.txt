cmake_minimum_required(VERSION 2.8)

project(glog_demo)

find_package(GLOG)

if(GLOG_FOUND)
message(STATUS "Found Glog.")
else(GLOG_FOUND)
message(FATAL "NOT Found Glog.")
endif(GLOG_FOUND)

include_directories(GLOG_INCLUDE_DIRS)

add_executable(glog_demo main.cpp)


target_link_libraries(glog_demo 
${GLOG_LIBRARY})
