cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

add_executable(base_test base_test.cpp)

target_link_libraries(base_test ${JSONCPP_LIBS})

