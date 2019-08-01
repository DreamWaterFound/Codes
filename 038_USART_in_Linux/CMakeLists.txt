cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(USART_DEMO)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


# 如果要使用 boost 这个库， 那么一定要这么写
find_package(Boost REQUIRED COMPONENTS system thread)

include_directories(${Boost_INCLUDE_DIRS})

link_directories(${Boost_LIBRARY_DIRS})

add_executable(usart_test ${CMAKE_SOURCE_DIR}/src/main.cpp)

target_link_libraries(usart_test ${Boost_LIBRARIES})

