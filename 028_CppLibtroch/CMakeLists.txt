cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# 最好是使用CUDA9，因为目前自己的电脑上只有在CUDA9下才安装了cuDNN

project(CppLibtorch)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(Torch REQUIRED)

add_executable(example_01 ${PROJECT_SOURCE_DIR}/src/example_01.cpp)
target_link_libraries(example_01 ${TORCH_LIBRARIES})
set_property(TARGET example_01 PROPERTY CXX_STANDARD 11)


