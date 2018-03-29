# CMake版本限制
cmake_minimum_required(VERSION 2.8)
# 工程名字
project( processingImage )
# 找opencv
find_package( OpenCV REQUIRED )
# 包含opencv
include_directories( ${OpenCV_INCLUDE_DIRS} )

#编译项目文件 平滑图像
add_executable( smoothingimg smoothing.cpp )
target_link_libraries(smoothingimg ${OpenCV_LIBS})

# 形态学操作 腐蚀erosion  膨胀dilation
add_executable( eroding_dilating shape_oper_eroding_dilating.cpp )
target_link_libraries(eroding_dilating ${OpenCV_LIBS})

# 形态学操作
# 开运算 (Opening)
# 闭运算 (Closing)
# 形态梯度 (Morphological Gradient)
# 顶帽 (Top Hat)
# 黑帽(Black Hat)
add_executable(morphology_Trans morphology_Transformations.cpp)
target_link_libraries(morphology_Trans ${OpenCV_LIBS})

# 形态学操作 腐蚀erosion  膨胀dilation 获取 水平线和垂直线
add_executable(extract_horizontal_vertical extract_horizontal_vertical_lines.cpp)
target_link_libraries(extract_horizontal_vertical ${OpenCV_LIBS})

# 图像金字塔
add_executable(pyramid pyramid.cpp)
target_link_libraries(pyramid ${OpenCV_LIBS})

# 阈值操作
# 0: Binary          二值
# 1: Binary Inverted 二值反
# 2: Threshold Truncated 截断
# 3: Threshold to Zero   阈值化为0 大于阈值的 保持原来的值 其余设置为0
#  4: Threshold to Zero Inverted  大于阈值的 设置为0  其余保持原来的值 
add_executable(threshold threshold.cpp)
target_link_libraries(threshold ${OpenCV_LIBS})


# 自定义滤波核 卷积操作
add_executable(linear_filter linear_filter2D.cpp)
target_link_libraries(linear_filter ${OpenCV_LIBS})


# 给图像添加边界
add_executable(copyMakeBorder copyMakeBorder.cpp)
target_link_libraries(copyMakeBorder ${OpenCV_LIBS})

# Sobel 算子 滤波器和 计算 水平垂直倒数后 再计算合成导数
add_executable(sobel_ker Sobel_ker.cpp)
target_link_libraries(sobel_ker ${OpenCV_LIBS})







