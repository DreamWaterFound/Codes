cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cmake_advance_test)

set(CMAKE_BUILD_TYPE Release)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 这个一定要定义在下面的配置文件之前
set(DATA_NUM 3.14159265)

option(MY_FLAG "A flag used." OFF)

configure_file(
    "${CMAKE_SOURCE_DIR}/config/Config.h.in"
    "${CMAKE_SOURCE_DIR}/inc/Config.h"
)

set(TEST_VAR "")





include_directories(${CMAKE_SOURCE_DIR}/inc)

add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/main.cc)

# 下面就是高级用法

# 1、在构建目标完成后，将生成的一些文件复制到指定的目录中
# 这种方式是在make阶段复制
add_custom_command(
    # 但是我没有搞明白为什么使用这个OUTPUT不可以
    # OUTPUT  ${CMAKE_SOURCE_DIR}/output/dummy.hpp
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND cp ${CMAKE_SOURCE_DIR}/inc/dummy.hpp ${CMAKE_SOURCE_DIR}/output/dummy_other.hpp 
    COMMAND echo "Running ..."
    COMMENT "Copying files ..."
)
# 另外一种方式：亲测也是完全可以的，不过这个是在cmake阶段生成
file(COPY ${CMAKE_SOURCE_DIR}/inc/dummy.hpp
     DESTINATION ${CMAKE_SOURCE_DIR}/output
)

# 2、通过自己搜索的方式获取所有源文件的列表
message(STATUS "==> All src files:")
file(GLOB_RECURSE SRC_LIST ${CMAKE_SOURCE_DIR}/src/ "*.cc") 
# 这里是通过输出来确定一下，列表中的文件都是否正确
foreach(src ${SRC_LIST})
    message(STATUS ${src})
endforeach(src ${SRC_LIST})
message(STATUS "Done.")

# 3、生成文件
set(LAUNCH_BATCH ${CMAKE_BINARY_DIR}/rundemo.sh)
file(WRITE ${LAUNCH_BATCH} ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} " ARGS")
# 调整权限，这里通过复制的方式
# message(STATUS "==> LAUNCH_BATCH = " ${LAUNCH_BATCH})
# file(COPY 
#         ${LAUNCH_BATCH} 
#     DESTINATION ${CMAKE_SOURCE_DIR} 
#     # 这里的权限参考 https://cmake.org/cmake/help/v3.6/command/install.html?highlight=file_permissions
#     # 这里是增加全局的可执行权限
#     NO_SOURCE_PERMISSIONS
#     # 但是好像只要添加了权限的操作，就会复制失败
#     FILE_PERMISSIONS GROUP_READ 
#     )# GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)     
add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND cp ${LAUNCH_BATCH} ${CMAKE_SOURCE_DIR}/rundemo.sh
    COMMAND chmod +x ${CMAKE_SOURCE_DIR}/rundemo.sh
    COMMENT "Generating bash file ..."
)




