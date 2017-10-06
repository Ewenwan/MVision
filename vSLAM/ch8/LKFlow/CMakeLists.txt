cmake_minimum_required( VERSION 2.8 )
project( useLK )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( useLK useLK.cpp )
target_link_libraries( useLK ${OpenCV_LIBS} )
