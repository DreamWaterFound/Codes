cmake_minimum_required(VERSION 2.8)

project(irrlicht_test)



## ===================================== BUILD ==================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_moudles)


## ==================================== PACKAGES ================================

# FIND CATKIN
find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport message_generation sensor_msgs)
# add the resized image message
add_message_files(DIRECTORY msg
   FILES ResizedImage.msg
)
generate_messages(DEPENDENCIES sensor_msgs)
catkin_package(CATKIN_DEPENDS cv_bridge image_transport message_runtime sensor_msgs)
include_directories(include ${catkin_INCLUDE_DIRS})

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



## =================================== WITH ROS ====================================
message(STATUS "Building without ROS .")
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME}
src/main_withROS.cpp
)

add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}
${catkin_LIBRARIES}
${IRRLICHT_LIBRARIES}
${OPENGL_LIBRARIES}
${AUX_LIBRARIES}
${OpenCV_LIBS}
)

install(TARGETS ${PROJECT_NAME}
ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(FILES resized_plugins.xml
DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)





