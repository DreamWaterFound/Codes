cmake_minimum_required( VERSION 2.8 )
project ( myslam )

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

set(G2O_ROOT "/home/guoqing/libraries/g2o_2010_eigen_3210")

############### dependencies ######################
# Eigen
include_directories( "/home/guoqing/libraries/eigen_3210/include/eigen3" )
# OpenCV
find_package( OpenCV 3.1 REQUIRED )
message(STATUS "==> OpenCV found: " ${OpenCV_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )
# Sophus 
find_package( Sophus REQUIRED )
message(STATUS "==> Sophus found: " ${Sophus_INCLUDE_DIRS})
include_directories( ${Sophus_INCLUDE_DIRS} )
link_directories( ${Sophus_LIBRARY_DIRS} )
# G2O
find_package( G2O REQUIRED )
message(STATUS "==> g2o found: " ${G2O_INCLUDE_DIRS} ${G2O_INCLUDE_DIR})
include_directories( ${G2O_INCLUDE_DIRS} ${G2O_INCLUDE_DIR})
# link_directories()

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_TYPES_SBA}
    # g2o_core g2o_stuff g2o_types_sba
)
############### dependencies ######################
include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory( src )
add_subdirectory( test )