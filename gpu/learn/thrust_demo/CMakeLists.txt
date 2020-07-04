cmake_minimum_required(VERSION 3.10)

find_package(CUDA REQUIRED)

if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -G -DTHRUST_DEBUG")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -g")
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda")

include_directories(/usr/local/cuda/samples/common/inc/)

file( GLOB APP_SOURCES "*.cu" "*/*.cu")
foreach( sourcefile ${APP_SOURCES} )
    string( REGEX REPLACE ".+/" "" executablename1 ${sourcefile} )
    string( REPLACE ".cu" "" executablename ${executablename1} )

    message ("source files: " ${sourcefile} " executable: " ${executablename})
    cuda_add_executable( ${executablename} ${sourcefile} )
endforeach( sourcefile ${APP_SOURCES} )