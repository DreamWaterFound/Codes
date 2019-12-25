cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


project(condition_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(Threads REQUIRED)

add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/main.cc )
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
