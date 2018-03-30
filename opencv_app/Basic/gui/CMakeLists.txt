# CMake版本限制
cmake_minimum_required(VERSION 2.8)
# 工程名字
project( gui )
# 找opencv
find_package( OpenCV REQUIRED )
# 包含opencv
include_directories( ${OpenCV_INCLUDE_DIRS} )

# 为程序界面添加滑动条
add_executable( trackbar_img_add trackbar_img_add.cpp)
target_link_libraries( trackbar_img_add ${OpenCV_LIBS} )




