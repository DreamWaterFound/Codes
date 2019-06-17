cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(Threads REQUIRED)

find_package(PythonLibs REQUIRED)

# OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("-- ==> Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")

if(PYTHONLIBS_FOUND)
    message(STATUS "Python Lib Found! Are: " ${PYTHON_LIBRARIES})
    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(${PROJECT_NAME} ./main.cpp)
    target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS} dl util)
else(PYTHONLIBS_FOUND)
    message(STATUS "Python Lib NOT FOUND!")
endif(PYTHONLIBS_FOUND)