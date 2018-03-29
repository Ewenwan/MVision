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

# Laplacian 算子 的离散模拟。 图像二阶倒数  梯度的梯度 0值的话 边缘概率较大
# 利用 Sobel 算子 实现图像二阶导数
add_executable(laplacian_ker laplacian.cpp)
target_link_libraries(laplacian_ker ${OpenCV_LIBS})

# candy边缘检测  消除噪声  计算梯度幅值和方向   非极大值 抑制  滞后阈值
add_executable(canny_Edge canny_Edge_Detector.cpp)
target_link_libraries(canny_Edge ${OpenCV_LIBS})


# 霍夫线变换 检测图像中的直线. 先candy边缘检测 再找直线
add_executable(hough_Line hough_Line_Transform.cpp)
target_link_libraries(hough_Line ${OpenCV_LIBS})

# 霍夫圆变换 检测图像中的圆. 先candy边缘检测 再找圆
add_executable(hough_circle hough_Circle_Transform.cpp)
target_link_libraries(hough_circle ${OpenCV_LIBS})

# 图像像素重映射 remapping
add_executable(remapping remapping.cpp)
target_link_libraries(remapping ${OpenCV_LIBS})

# 随机矩阵空间映射  仿射变换 正交单位阵 欧式变化   正交阵 相似变换  非正交阵 仿射变换
add_executable(affine_Transf affine_Transformations.cpp)
target_link_libraries(affine_Transf ${OpenCV_LIBS})

# 求得对直方图均衡化的映射矩阵 在对原图像进行映射
add_executable(histogram_Equa histogram_Equalization.cpp)
target_link_libraries(histogram_Equa ${OpenCV_LIBS})

# 图像直方图计算 
add_executable(histogram_Calc histogram_Calculation.cpp)
target_link_libraries(histogram_Calc ${OpenCV_LIBS})

# 反向投影   利用直方图模型 搜索 对应的 图像区域
add_executable(back_Projection back_Projection.cpp)
target_link_libraries(back_Projection ${OpenCV_LIBS})



