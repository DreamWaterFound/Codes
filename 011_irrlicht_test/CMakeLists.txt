cmake_minimum_required(VERSION 2.8)

project(irrlicht_test)



## ===================================== BUILD ==================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_moudles)


## ==================================== PACKAGES ================================

# FIND IRRLICHT
find_package(Irrlicht REQUIRED)
if(IRRLICHT_FOUND)
  message(STATUS " ==>> GQ : Irrlicht found! lib dir = " ${IRRLICHT_INCLUDE_DIR})
  include_directories(${IRRLICHT_INCLUDE_DIR})
  # NEED IT
  list(APPEND AUX_LIBRARIES
    /usr/lib/x86_64-linux-gnu/libX11.so
    /usr/lib/x86_64-linux-gnu/libXxf86vm.so.1
  )
endif(IRRLICHT_FOUND)

# FIND OPENGL
find_package(OpenGL REQUIRED)

# FIND OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("-- ==> Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")



## =================================== WITHOUT ROS ====================================
message(STATUS "Building without ROS .")
add_executable(${PROJECT_NAME}
src/main_withoutROS.cpp
)

target_link_libraries(${PROJECT_NAME}
${IRRLICHT_LIBRARIES}
${OPENGL_LIBRARIES}
${AUX_LIBRARIES}
${OpenCV_LIBS}
)




