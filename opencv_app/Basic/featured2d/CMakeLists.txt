# CMake版本限制
cmake_minimum_required(VERSION 2.8)
# 工程名字
project( feature2d )
# 找opencv
find_package( OpenCV REQUIRED )
# 包含opencv
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Harris角点检测
# 计算 R = det(M)  -  k*(trace(M))^2   的图像矩阵 看是否大于阈值
add_executable( harris_corner harris_corner_detector.cpp)
target_link_libraries( harris_corner ${OpenCV_LIBS} )

# Shi-Tomasi 角点检测 算法 是Harris 算法的改进
# mini(lamd1,lamd2) > 阈值 认为是角点
add_executable(shi_Tomasi shi_Tomasi_corner_detector.cpp)
target_link_libraries( shi_Tomasi ${OpenCV_LIBS} )

# FAST角点检测算法  ORB特征检测中使用的就是这种角点检测算法
# 若某像素与其周围邻域内足够多的像素点相差较大，则该像素可能是角点。
add_executable(fast_corner fast_corner_detector.cpp)
target_link_libraries( fast_corner ${OpenCV_LIBS} )

# 通过自定义 R的计算方法和自适应阈值 来定制化检测角点
add_executable(creating_corner creating_corner_detector.cpp)
target_link_libraries( creating_corner ${OpenCV_LIBS} )

# 亚像素级的角点检测
add_executable(subpixeles_corner subpixeles_corner.cpp)
target_link_libraries( subpixeles_corner ${OpenCV_LIBS} )



