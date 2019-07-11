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
    include_directories(${PROJECT_SOURCE_DIR}/include)
    include_directories(${PROJECT_SOURCE_DIR})
    include_directories(BEFORE "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/my_py_pt/lib/python3.7/site-packages/numpy/core/include/numpy")
    include_directories(BEFORE "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/my_py_pt/lib/python3.7/site-packages/numpy/core/include")

    

    add_library(yolact yolact.cpp)
    target_link_libraries(yolact ${PYTHON_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS} dl util)

    add_executable(yolact_test ./main.cpp)
    target_link_libraries(yolact_test yolact )

    add_executable(tum_yolact_test ./tum_test.cpp)
    target_link_libraries(tum_yolact_test yolact )

    add_dependencies(yolact_test yolact)
    add_dependencies(tum_yolact_test yolact)


else(PYTHONLIBS_FOUND)
    message(STATUS "Python Lib NOT FOUND!")
endif(PYTHONLIBS_FOUND)