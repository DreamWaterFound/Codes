cmake_minimum_required(VERSION 2.8)
# Project Name
PROJECT(OpenGL_Picker)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


#########################################################
# FIND GLUT
#########################################################
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)
#########################################################
# FIND OPENGL
#########################################################
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)
#########################################################
# Include Files
#########################################################
add_executable(my_test_01 src/my_test_01.cpp)
add_executable(my_test_02 src/my_test_02.cpp)
add_executable(my_test_03 src/my_test_03.cpp)
add_executable(my_test_04 src/my_test_04.cpp)
add_executable(my_test_05 src/my_test_05.cpp)



########################################################
# Linking & stuff
#########################################################

# create the program "test"
target_link_libraries(my_test_01 ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
target_link_libraries(my_test_02 ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
target_link_libraries(my_test_03 ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
target_link_libraries(my_test_04 ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
target_link_libraries(my_test_05 ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )


