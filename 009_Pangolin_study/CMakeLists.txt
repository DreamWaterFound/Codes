cmake_minimum_required(VERSION 2.8)

project(Panglion_Study)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


# Eigen
find_package(Eigen 3 REQUIRED)
message("-- ==> Found Eigen ${EIGEN3_VERSION} in ${EIGEN_INCLUDE_DIR}")
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR}/..)


find_package(Pangolin REQUIRED)
message("-- ==> Found Pangolin ${Pangolin_VERSION} in ${Pangolin_INCLUDE_DIRS}")
include_directories(${Pangolin_INCLUDE_DIRS})


find_package(OpenCV 3 REQUIRED)
message("-- ==> Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(00_project_test src/00_project_test.cpp)

add_executable(01_HelloPangolin src/01_HelloPangolin.cpp)
target_link_libraries(01_HelloPangolin ${Pangolin_LIBRARIES})

add_executable(02_HelloPangolinOffscreen src/02_HelloPangolinOffscreen.cpp)
target_link_libraries(02_HelloPangolinOffscreen ${Pangolin_LIBRARIES})

add_executable(03_HelloPangolinThreads src/03_HelloPangolinThreads.cpp)
target_link_libraries(03_HelloPangolinThreads ${Pangolin_LIBRARIES})

add_executable(04_SharedMemoryCamera src/04_SharedMemoryCamera.cpp)
target_link_libraries(04_SharedMemoryCamera ${Pangolin_LIBRARIES})

add_executable(05_SimpleDisplay src/05_SimpleDisplay.cpp)
target_link_libraries(05_SimpleDisplay ${Pangolin_LIBRARIES})

add_executable(06_SimpleDisplayImage src/06_SimpleDisplayImage.cpp)
target_link_libraries(06_SimpleDisplayImage ${Pangolin_LIBRARIES})

add_executable(07_SimpleMultiDisplay src/07_SimpleMultiDisplay.cpp)
target_link_libraries(07_SimpleMultiDisplay ${Pangolin_LIBRARIES})

add_executable(08_SimplePlot src/08_SimplePlot.cpp)
target_link_libraries(08_SimplePlot ${Pangolin_LIBRARIES})

add_executable(09_SimpleRecord src/09_SimpleRecord.cpp)
target_link_libraries(09_SimpleRecord ${Pangolin_LIBRARIES})

add_executable(10_SimpleScene src/10_SimpleScene.cpp)
target_link_libraries(10_SimpleScene ${Pangolin_LIBRARIES})

add_executable(11_SimpleVideo src/11_SimpleVideo.cpp)
target_link_libraries(11_SimpleVideo ${Pangolin_LIBRARIES})

# 需要使用cuda，但是本机目前没有这个环境
# add_executable(12_VBODisplay src/12_VBODisplay.cpp)
# target_link_libraries(12_VBODisplay ${Pangolin_LIBRARIES})


## ========================= SELF TESTS ==============================
add_executable(test1 test/test1.cpp)
target_link_libraries(test1 ${Pangolin_LIBRARIES})

add_executable(test2 test/test2.cpp)
target_link_libraries(test2 ${Pangolin_LIBRARIES})

add_executable(test3 test/test3.cpp)
target_link_libraries(test3 ${Pangolin_LIBRARIES})

add_executable(test4 test/test4.cpp)
target_link_libraries(test4 ${Pangolin_LIBRARIES})

add_executable(test5 test/test5.cpp)
target_link_libraries(test5 ${Pangolin_LIBRARIES})

add_executable(test6 test/test6.cpp)
target_link_libraries(test6 ${Pangolin_LIBRARIES})

add_executable(test7 test/test7.cpp)
target_link_libraries(test7 ${Pangolin_LIBRARIES} ${OpenCV_LIBS})

add_executable(test8 test/test8.cpp)
target_link_libraries(test8 ${Pangolin_LIBRARIES} ${OpenCV_LIBS})

add_executable(test9 test/test9.cpp)
target_link_libraries(test9 ${Pangolin_LIBRARIES} ${OpenCV_LIBS})

add_executable(test0 test/test0.cpp)
target_link_libraries(test0 ${Pangolin_LIBRARIES} ${OpenCV_LIBS})