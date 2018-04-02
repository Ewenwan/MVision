# CMake版本限制
cmake_minimum_required(VERSION 2.8)
# 工程名字
project( video_proc )
# 找opencv
find_package( OpenCV REQUIRED )
# 包含opencv
include_directories( ${OpenCV_INCLUDE_DIRS} )

# 背景去除
add_executable( backGround_sub background_Subtraction.cpp)
target_link_libraries( backGround_sub ${OpenCV_LIBS} )



