#版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( imageBasics )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )


# 添加可执行文件
add_executable( imageBasics imageBasics.cpp )
# 链接OpenCV库
target_link_libraries( imageBasics ${OpenCV_LIBS} )

# 添加可执行文件
add_executable( bleblending_two  blending_two_images.cpp)
# 链接OpenCV库
target_link_libraries( bleblending_two ${OpenCV_LIBS} )
