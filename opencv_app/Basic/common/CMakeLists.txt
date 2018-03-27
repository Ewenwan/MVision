# CMake版本限制
cmake_minimum_required(VERSION 2.8)
# 工程名字
project( DisplayImage )
# 找opencv
find_package( OpenCV REQUIRED )
# 包含opencv
include_directories( ${OpenCV_INCLUDE_DIRS} )

#编译项目文件 显示图片
add_executable( displayImage displayImage.cpp )
target_link_libraries( displayImage ${OpenCV_LIBS} )

# 载入修改保存图片
add_executable( modiImage load__modi_save_img.cpp)
target_link_libraries( modiImage ${OpenCV_LIBS} )
# 画线 椭圆
add_executable(drawing  Drawing_1.cpp)
target_link_libraries( drawing ${OpenCV_LIBS} )
#随机画线 椭圆
add_executable(randomdrawing  Drawing_2.cpp)
target_link_libraries( randomdrawing ${OpenCV_LIBS} )

# 访问元素
#【1】C语言的下标[]访问
#【2】迭代器访问 MatIterator_<> it, end;
#【3】at<>()随机访问
#【4】内置LUT函数访问 方式效率最高
add_executable( scan_images how_to_scan_images.cpp)
target_link_libraries( scan_images ${OpenCV_LIBS} )

