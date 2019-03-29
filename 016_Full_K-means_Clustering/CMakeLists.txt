cmake_minimum_required(VERSION 2.8)

project(K_means_demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_moudles)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR})

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

# ============ demos ==================

add_executable(demo_k_mean_2D_points    ${PROJECT_SOURCE_DIR}/src/demos/demo_k_mean_2D_points.cpp)
target_link_libraries(demo_k_mean_2D_points ${OpenCV_LIBS})

add_executable(demo_k_mean_RGB_pic    ${PROJECT_SOURCE_DIR}/src/demos/demo_k_mean_RGB_pic.cpp)
target_link_libraries(demo_k_mean_RGB_pic ${OpenCV_LIBS})

add_executable(demo_k_mean_RGB_uv_pic    ${PROJECT_SOURCE_DIR}/src/demos/demo_k_mean_RGB_uv_pic.cpp)
target_link_libraries(demo_k_mean_RGB_uv_pic ${OpenCV_LIBS})

add_executable(demo_K_mean_RGBD_uv_pic    ${PROJECT_SOURCE_DIR}/src/demos/demo_K_mean_RGBD_uv_pic.cpp)
target_link_libraries(demo_K_mean_RGBD_uv_pic ${OpenCV_LIBS})


# ============= tools =================
add_executable(create2DSamples
    ${PROJECT_SOURCE_DIR}/src/tools/create2DSamples.cpp
)
target_link_libraries(create2DSamples ${OpenCV_LIBS})


add_executable(see2DSamples
    ${PROJECT_SOURCE_DIR}/src/tools/see2DSamples.cpp
)
target_link_libraries(see2DSamples ${OpenCV_LIBS})