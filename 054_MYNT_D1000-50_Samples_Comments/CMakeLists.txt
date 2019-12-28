cmake_minimum_required(VERSION 3.0)

project(MYNT_EYE_D_Samples) #  VERSION 1.3.6 LANGUAGES C CXX)

# 设置编译器标志
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native")

# 搜索OpenCV
set(OpenCV_DIR "/home/guoqing/libraries/opencv_331/share/OpenCV")
find_package(OpenCV 3 REQUIRED)
message(STATUS "==> OpenCV 3: OpenCV_INCLUDE_DIRS = " ${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND LINK_LIBS ${OpenCV_LIBS})

# 搜索 PCL
set(PCL_DIR "/home/guoqing/libraries/pcl_172/share/pcl-1.7")
find_package(PCL REQUIRED)
message(STATUS "==> PCL: PCL_INCLUDE_DIRS = " ${PCL_INCLUDE_DIRS})
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )
list(APPEND LINK_LIBS ${PCL_LIBRARIES})


# 搜索小觅相机驱动
set(mynteyed_DIR "/home/guoqing/libraries/MYNT_EYE_D_180/lib/cmake/mynteyed")
find_package(mynteyed REQUIRED)
message(STATUS "==> Found mynteye: ${mynteyed_VERSION}")
list(APPEND LINK_LIBS mynteye_depth)

if(mynteyed_WITH_OPENCV)
  add_definitions(-DWITH_OPENCV)
  message(STATUS "mynteye with opencv.")
endif()

# 包含文件设置
include_directories(${CMAKE_SOURCE_DIR}/src)


# 生成文件
add_executable(get_image
    ${CMAKE_SOURCE_DIR}/src/get_image.cc
    ${CMAKE_SOURCE_DIR}/src/util/cam_utils.cc
    ${CMAKE_SOURCE_DIR}/src/util/cv_painter.cc
)

target_link_libraries(get_image ${LINK_LIBS})

add_executable(get_stereo_image
    ${CMAKE_SOURCE_DIR}/src/get_stereo_image.cc
    ${CMAKE_SOURCE_DIR}/src/util/cam_utils.cc
    ${CMAKE_SOURCE_DIR}/src/util/cv_painter.cc
)

target_link_libraries(get_stereo_image ${LINK_LIBS})

add_executable(get_depth
    ${CMAKE_SOURCE_DIR}/src/get_depth.cc
    ${CMAKE_SOURCE_DIR}/src/util/cam_utils.cc
    ${CMAKE_SOURCE_DIR}/src/util/cv_painter.cc
)

target_link_libraries(get_depth ${LINK_LIBS})

add_executable(get_depth_with_filter
    ${CMAKE_SOURCE_DIR}/src/get_depth_with_filter.cc
    ${CMAKE_SOURCE_DIR}/src/util/cam_utils.cc
    ${CMAKE_SOURCE_DIR}/src/util/cv_painter.cc
)

target_link_libraries(get_depth_with_filter ${LINK_LIBS})

add_executable(get_imu
    ${CMAKE_SOURCE_DIR}/src/get_imu.cc
    ${CMAKE_SOURCE_DIR}/src/util/cam_utils.cc
    ${CMAKE_SOURCE_DIR}/src/util/cv_painter.cc
)

target_link_libraries(get_imu ${LINK_LIBS})

add_executable(get_from_callbacks
    ${CMAKE_SOURCE_DIR}/src/get_from_callbacks.cc
    ${CMAKE_SOURCE_DIR}/src/util/cam_utils.cc
    ${CMAKE_SOURCE_DIR}/src/util/cv_painter.cc
)

target_link_libraries(get_from_callbacks ${LINK_LIBS})


add_executable(get_all_with_options
    ${CMAKE_SOURCE_DIR}/src/get_all_with_options.cc
    ${CMAKE_SOURCE_DIR}/src/util/cam_utils.cc
    ${CMAKE_SOURCE_DIR}/src/util/cv_painter.cc
)

target_link_libraries(get_all_with_options ${LINK_LIBS})

add_executable(get_img_params
    ${CMAKE_SOURCE_DIR}/src/get_img_params.cc
)
target_link_libraries(get_img_params ${LINK_LIBS})

add_executable(get_imu_params
    ${CMAKE_SOURCE_DIR}/src/get_imu_params.cc
)
target_link_libraries(get_imu_params ${LINK_LIBS})


add_executable(get_gps
    ${CMAKE_SOURCE_DIR}/src/get_gps.cc
)
target_link_libraries(get_gps ${LINK_LIBS})


add_executable(get_ultrasonic
    ${CMAKE_SOURCE_DIR}/src/get_ultrasonic.cc
)
target_link_libraries(get_ultrasonic ${LINK_LIBS})

add_executable(auto_restart
    ${CMAKE_SOURCE_DIR}/src/auto_restart.cc
    ${CMAKE_SOURCE_DIR}/src/util/cam_utils.cc
    ${CMAKE_SOURCE_DIR}/src/util/cv_painter.cc
)
target_link_libraries(auto_restart ${LINK_LIBS})

# add_executable(auto_restart
#     ${CMAKE_SOURCE_DIR}/src/auto_restart.cc
#     ${CMAKE_SOURCE_DIR}/src/util/cam_utils.cc
#     ${CMAKE_SOURCE_DIR}/src/util/cv_painter.cc
# )
# target_link_libraries(auto_restart ${LINK_LIBS})

add_executable(get_points
    ${CMAKE_SOURCE_DIR}/src/get_points.cc
    ${CMAKE_SOURCE_DIR}/src/util/cam_utils.cc
    ${CMAKE_SOURCE_DIR}/src/util/cv_painter.cc
    ${CMAKE_SOURCE_DIR}/src/util/pc_viewer.cc
)
target_link_libraries(get_points ${LINK_LIBS})



add_executable(view_points
    ${CMAKE_SOURCE_DIR}/src/points/view_points.cc
)
target_link_libraries(view_points ${LINK_LIBS})

add_executable(get_points2
    ${CMAKE_SOURCE_DIR}/src//points/get_points2.cc
    ${CMAKE_SOURCE_DIR}/src/util/pc_utils.cc
)
target_link_libraries(get_points2 ${LINK_LIBS})

add_executable(get_points_with_filter
    ${CMAKE_SOURCE_DIR}/src/points/get_points_with_filter.cc
    ${CMAKE_SOURCE_DIR}/src/util/pc_utils.cc
)
target_link_libraries(get_points_with_filter ${LINK_LIBS})



