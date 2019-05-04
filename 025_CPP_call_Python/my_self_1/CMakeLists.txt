cmake_minimum_required(VERSION 2.8)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

project(python_call_my_test)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



find_package(PythonLibs REQUIRED)

if(PYTHONLIBS_FOUND)
    message(STATUS "Python Lib Found! Are: " ${PYTHON_LIBRARIES})
else(PYTHONLIBS_FOUND)
    message(STATUS "Python Lib NOT FOUND!")
endif(PYTHONLIBS_FOUND)

include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ./main.cpp)

target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})