cmake_minimum_required(VERSION 2.8)

project(Accelerate_test)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# SSE support
if (SSE3_FOUND)
    message(STATUS " ==>> SSE3 Detected.")
    set (CMAKE_C_FLAGS_ALL "${CMAKE_C_FLAGS_ALL} -msse3")
endif()
if (SSSE3_FOUND)
    message(STATUS " ==>> SSSE3 Detected.")
    set (CMAKE_C_FLAGS_ALL "${CMAKE_C_FLAGS_ALL} -mssse3")
endif()
if (SSE4_1_FOUND)
    message(STATUS " ==>> SSE4.1 Detected.")
    set (CMAKE_C_FLAGS_ALL "${CMAKE_C_FLAGS_ALL} -msse4.1")
endif()
if (SSE4_2_FOUND)
    message(STATUS " ==>> SSE4.2 Detected.")
    set (CMAKE_C_FLAGS_ALL "${CMAKE_C_FLAGS_ALL} -msse4.2")
endif()




add_executable(generateData ${PROJECT_SOURCE_DIR}/tools/generateData.cc)

add_executable(SSE_demo ${PROJECT_SOURCE_DIR}/src/SSE_Demo.cc)

