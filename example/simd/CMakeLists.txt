add_definitions(-mavx)

file( GLOB APP_SOURCES "*.cpp" "./*/*.cpp")
foreach( sourcefile ${APP_SOURCES} )
    # executable name cut off .cpp
    string( REGEX REPLACE ".+/" "simd_" executablename1 ${sourcefile} )
    string( REPLACE ".cpp" "" executablename ${executablename1} )

    message ("source files: " ${sourcefile} " executable: " ${executablename})
    add_executable( ${executablename} ${sourcefile} )
    target_link_libraries( ${executablename} )
endforeach( sourcefile ${APP_SOURCES} )
