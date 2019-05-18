cmake_minimum_required(VERSION 2.8)

project(QR_SOLVER)

# OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("-- ==> Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")
list(APPEND LINK_LIBS ${OpenCV_LIBS})

add_executable(${PROJECT_NAME} qr.cc)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})