cmake_minimum_required(VERSION 2.8)

project(Google_test)

# 首先是正常的设置

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
include_directories(${PROJECT_SOURCE_DIR}/inc)

# 寻找包
# TODO 换成DIR行不行？
set(GTEST_ROOT "/home/guoqing/libraries/gtest1100")
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_ROOT}/lib)
# set(GTEST_LLIB )
message(STATUS "==> GTEST_INCLUDE_DIR: " ${GTEST_INCLUDE_DIR})
message(STATUS "==> GTEST_LIBRARIES: " ${GTEST_LIBRARIES})

find_package(Threads)



# 自己的库
set(MYLIB "Add")
add_library(${MYLIB} ${CMAKE_SOURCE_DIR}/src/Add.cc)

# demo
add_executable(demo ${CMAKE_SOURCE_DIR}/src/main.cc)
target_link_libraries(demo ${MYLIB})


# 测试  
add_executable(normalTest ${CMAKE_SOURCE_DIR}/test/NormalTest.cc)
target_link_libraries(normalTest ${MYLIB} ${GTEST_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )#gtest gtest_main gmock gmock_main)

ENABLE_TESTING()
add_test(normal1 ../bin/normalTest)



