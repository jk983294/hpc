cmake_minimum_required(VERSION 3.5)

project(hpc)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wall ")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -Wall ")
message ("cxx Flags: " ${CMAKE_CXX_FLAGS})

# Source code
include_directories("/home/$USER/include")
include_directories("src")

add_subdirectory(example)
add_subdirectory(bench)
