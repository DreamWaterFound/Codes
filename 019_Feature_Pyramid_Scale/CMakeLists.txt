cmake_minimum_required(VERSION 2.8)

project(Feature_Pyramid_Scale)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("-- ==> Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")
list(APPEND LINK_LIBS ${OpenCV_LIBS})

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cc)

target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})



