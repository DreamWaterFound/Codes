cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(serialization_demo)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(Boost REQUIRED COMPONENTS serialization)

include_directories(${Boost_INCLUDE_DIRS})

add_executable(serialization_demo src/main.cc)

target_link_libraries(serialization_demo ${Boost_LIBRARIES})
