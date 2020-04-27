cmake_minimum_required(VERSION 2.8)

project(demo)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g -Wall -Wformat")

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)
list(APPEND LINK_LIBS ${OPENGL_LIBRARIES} GL rt dl)

# 这里按需修改
set(glfw3_DIR "${PROJECT_SOURCE_DIR}/ThirdParty/glfw-3.3.2/install/lib/cmake/glfw3")
find_package(glfw3 REQUIRED)

# 这里按需修改

set(gl3w_DIR "${PROJECT_SOURCE_DIR}/ThirdParty/gl3w/install/share/gl3w")
find_package(gl3w REQUIRED)
include_directories("${PROJECT_SOURCE_DIR}/ThirdParty/gl3w/install/include/gl3w")

# add_library(gl3w_lib ${PROJECT_SOURCE_DIR}/ThirdParty/gl3w/install/share/gl3w/gl3w.c)



# 好像是不需要添加头文件， 库的话直接写 glfw 就行了

# include_directories(
#     ${PROJECT_SOURCE_DIR}/../ThirdParty/imgui_1.75/inc
#     ${PROJECT_SOURCE_DIR}/../ThirdParty/imgui_1.75/gl3w
# )

# add_executable(test_demo
#     # main.cpp
#     myTest.cc
#     ${PROJECT_SOURCE_DIR}/../ThirdParty/imgui_1.75/src/imgui_impl_glfw.cpp 
#     ${PROJECT_SOURCE_DIR}/../ThirdParty/imgui_1.75/src/imgui_impl_opengl3.cpp 
#     ${PROJECT_SOURCE_DIR}/../ThirdParty/imgui_1.75/src/imgui.cpp 
#     ${PROJECT_SOURCE_DIR}/../ThirdParty/imgui_1.75/src/imgui_demo.cpp 
#     ${PROJECT_SOURCE_DIR}/../ThirdParty/imgui_1.75/src/imgui_draw.cpp 
#     ${PROJECT_SOURCE_DIR}/../ThirdParty/imgui_1.75/src/imgui_widgets.cpp 
#     ${PROJECT_SOURCE_DIR}/../ThirdParty/imgui_1.75/gl3w/GL/gl3w.c
# )

add_executable(test_demo
    ${PROJECT_SOURCE_DIR}/src/main.cc
)

target_link_libraries(test_demo GL rt dl glfw gl3w)

