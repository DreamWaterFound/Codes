cmake_minimum_required(VERSION 2.8)

project(Shared_memory_test)

set(CUDA_NVCC_FLAGS -O3;-G;-g)

find_package(CUDA)

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("-- ==> Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")
list(APPEND LINK_LIBS ${OpenCV_LIBS})


cuda_add_library(${PROJECT_NAME} SHARED kernel.cu)

add_executable(SMem_test main.cc)

target_link_libraries(SMem_test ${PROJECT_NAME} ${LINK_LIBS})

