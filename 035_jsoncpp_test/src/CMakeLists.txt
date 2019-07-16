cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

add_executable(base_test base_test.cpp)
add_executable(yolact_result_parse yolact_result_parse.cpp)

target_link_libraries(base_test ${JSONCPP_LIBS})
target_link_libraries(yolact_result_parse ${JSONCPP_LIBS})

