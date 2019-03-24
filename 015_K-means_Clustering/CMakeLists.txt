cmake_minimum_required(VERSION 2.8)

project(K_means_demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_moudles)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Eigen
find_package(Eigen 3 REQUIRED)
message("-- ==> Found Eigen ${EIGEN3_VERSION} in ${EIGEN_INCLUDE_DIR}")
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR}/..)

# OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("-- ==> Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")
list(APPEND LINK_LIBS ${OpenCV_LIBS})

add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/src/main.cpp
)

add_executable(createSamples
    ${PROJECT_SOURCE_DIR}/tools/createSamples.cpp
)

add_executable(seeSamples
    ${PROJECT_SOURCE_DIR}/tools/seeSamples.cpp
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(createSamples ${OpenCV_LIBS})
target_link_libraries(seeSamples ${OpenCV_LIBS})