cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

add_executable(build_toml build_toml.cpp)

add_executable(conversions conversions.cpp)

add_executable(parse_stdin parse_stdin.cpp)

add_executable(parse parse.cpp)

add_executable(my_example my_example.cpp)