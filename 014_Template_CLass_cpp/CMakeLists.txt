cmake_minimum_required(VERSION 2.8)

project(TemplateTest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(${PROJECT_NAME} 
./src/main.cpp
./src/Tclass.cpp
./src/VClassSphere.cpp
./src/VClassVolumeBase.cpp
)
