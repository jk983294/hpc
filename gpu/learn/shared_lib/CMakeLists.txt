cmake_minimum_required(VERSION 3.10)

find_package(CUDA REQUIRED)

if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -G -DTHRUST_DEBUG")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -g")
endif()

cuda_add_library(test_static_lib STATIC a.cu)
set_target_properties(test_static_lib
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON)

cuda_add_executable( my_static_app main.cpp )
target_link_libraries(my_static_app test_static_lib)

cuda_add_library(test_dynamic_lib SHARED a.cu)
set_target_properties(test_dynamic_lib
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON)

cuda_add_executable( my_dynamic_app main.cpp )
target_link_libraries(my_dynamic_app test_dynamic_lib)