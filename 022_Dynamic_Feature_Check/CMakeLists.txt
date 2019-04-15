cmake_minimum_required(VERSION 2.8)

project(Dynamic_Feature_Check)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Realse)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Eigen
# find_package(Eigen 3 REQUIRED)
# message("-- ==> Found Eigen ${EIGEN3_VERSION} in ${EIGEN_INCLUDE_DIR}")
# include_directories(${EIGEN_INCLUDE_DIR})
# include_directories(${EIGEN_INCLUDE_DIR}/..)

# OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("-- ==> Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")
list(APPEND LINK_LIBS ${OpenCV_LIBS})

include_directories(${PROJECT_SOURCE_DIR}/include)


add_executable(${PROJECT_NAME}_v1 src/main_v1.cc)
target_link_libraries(${PROJECT_NAME}_v1 ${LINK_LIBS})

add_executable(${PROJECT_NAME}_v2 src/main_v2.cc)
target_link_libraries(${PROJECT_NAME}_v2 ${LINK_LIBS})
