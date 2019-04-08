cmake_minimum_required(VERSION 2.6)
project (VTK_ImageScale)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

if(NOT VTK_BINARY_DIR)
    find_package(VTK REQUIRED)
    # message(STATUS "VTK_USE_FILE = " ${VTK_USE_FILE})
    # message(STATUS "VTK_LIBRARIES = " ${VTK_LIBRARIES})
    include(${VTK_USE_FILE})
    # if(NOT VTK_USE_RENDERING)
        # message(FATAL_ERROR "${PROJECT_NAME} requires VTK_USE_RENDERING.")
    # endif(NOT VTK_USE_RENDERING)
    # include(${VTK_USE_FILE})
endif(NOT VTK_BINARY_DIR)

add_executable(demo_scaled_obj src/demo_scaled_obj.cc)
target_link_libraries(demo_scaled_obj ${VTK_LIBRARIES})

add_executable(demo_scale_image src/demo_scale_image.cc)
target_link_libraries(demo_scale_image ${VTK_LIBRARIES})

add_executable(demo_scale_image_offical src/demo_scale_image_offical.cc)
target_link_libraries(demo_scale_image_offical ${VTK_LIBRARIES})

