# CMake版本限制
cmake_minimum_required(VERSION 2.8)
# 工程名字
project( DisplayImage )
# 找opencv
find_package( OpenCV REQUIRED )
# 包含opencv
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable( single_tracker single_tracker.cpp )
target_link_libraries( single_tracker ${OpenCV_LIBS} )

add_executable( opticalFlow opticalFlow.cpp)
target_link_libraries( opticalFlow ${OpenCV_LIBS} )
