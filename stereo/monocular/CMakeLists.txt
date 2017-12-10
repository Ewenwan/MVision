#版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( Monocular_Calibr )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )

#由图片文件夹生成 图片文件路径 yaml文件
add_executable( imagelist_creator imagelist_creator.cpp)
# 链接OpenCV库
target_link_libraries(imagelist_creator  ${OpenCV_LIBS} )

#单目相机矫正程序
add_executable( Monocular_Calibr calibration.cpp )
# 链接OpenCV库
target_link_libraries( Monocular_Calibr  ${OpenCV_LIBS} )