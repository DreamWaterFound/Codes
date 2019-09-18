cmake_minimum_required( VERSION 2.8 )

project(pcl_segmentation_test)

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -Wno-deprecated-declarations" )

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(Eigen3_DIR "")
find_package(Eigen3 3.1.0 REQUIRED)
message(STATUS "Eigen3 Found: "  ${EIGEN3_INCLUDE_DIR})

# pcl 
set(PCL_DIR "/home/guoqing/libraries/pcl_172/share/pcl-1.7")
# find_package( PCL REQUIRED COMPONENT common io filters )
find_package(PCL REQUIRED)

include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

add_executable( pcl_segmentation_test ${PROJECT_SOURCE_DIR}/src/main.cc )
target_link_libraries( pcl_segmentation_test ${PCL_LIBRARIES} )