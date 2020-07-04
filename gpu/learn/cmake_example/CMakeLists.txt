cmake_minimum_required(VERSION 3.10)

find_package(CUDA REQUIRED)

include_directories(/usr/local/cuda/samples/common/inc/)

file( GLOB APP_SOURCES "*.cu")
foreach( sourcefile ${APP_SOURCES} )
    string( REGEX REPLACE ".+/" "" executablename1 ${sourcefile} )
    string( REPLACE ".cu" "" executablename ${executablename1} )

    message ("source files: " ${sourcefile} " executable: " ${executablename})
    cuda_add_executable( ${executablename} ${sourcefile} )
endforeach( sourcefile ${APP_SOURCES} )