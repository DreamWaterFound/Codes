cmake_minimum_required(VERSION 2.8)

project(python_call_test)

find_package(PythonLibs REQUIRED)

if(PYTHONLIBS_FOUND)
    message(STATUS "Python Lib Found! Are: " ${PYTHON_LIBRARIES})
else(PYTHONLIBS_FOUND)
    message(STATUS "Python Lib NOT FOUND!")
endif(PYTHONLIBS_FOUND)

include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ./py_call_test.cpp)

target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})