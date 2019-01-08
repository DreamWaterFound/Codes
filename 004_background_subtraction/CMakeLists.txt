cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11")

project(background_subtraction)

find_package(OpenCV 3 REQUIRED)

if(OPENCV_FOUND)

    message(STATUS "OpenCV Found!")
    message(STATUS "Include directory:" ${OpenCV_INCLUDE_DIRS})

    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${PROJECT_SOURCE_DIR}/include)

    add_executable(${PROJECT_NAME}
    src/main.cpp
    src/DataReader.cpp
    # src/MotionDetector.cpp
    src/MotionDetector_base.cpp
    src/MotionDetector_DiffBase.cpp
    src/MotionDetector_backsub.cpp
    src/MotionDetector_framesub.cpp
    src/MotionDetector_3feamesub.cpp
    src/MotionDetector_GMM.cpp
    )

    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

else(OPENCV_FOUND)
message(ERROR "OpenCV NOT Found!")
endif(OPENCV_FOUND)


