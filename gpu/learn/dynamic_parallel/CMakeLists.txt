cmake_minimum_required(VERSION 3.10)

SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "-arch=sm_61")

get_filename_component(CUDA_LIB_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY)
find_library(CUDA_cudadevrt_LIBRARY cudadevrt PATHS ${CUDA_LIB_PATH})

include_directories(/usr/local/cuda/samples/common/inc/)

file( GLOB APP_SOURCES "*.cu")
foreach( sourcefile ${APP_SOURCES} )
    string( REGEX REPLACE ".+/" "" executablename1 ${sourcefile} )
    string( REPLACE ".cu" "" executablename ${executablename1} )

    message ("source files: " ${sourcefile} " executable: " ${executablename})
    cuda_add_executable( ${executablename} ${sourcefile} )
    target_link_libraries(${executablename} ${CUDA_cudadevrt_LIBRARY})
endforeach( sourcefile ${APP_SOURCES} )