#版本
cmake_minimum_required( VERSION 2.8 )
#工程名
project( vo1 )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# 找 opencv'库 3.1
find_package( OpenCV 3.1 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 
#找 G2O库
find_package( G2O REQUIRED )
#找 CSparse 库  稀疏矩阵求解库    稀疏雅克比矩阵  海塞矩阵 求解
find_package( CSparse REQUIRED )

# 包含库
include_directories( 
    ${OpenCV_INCLUDE_DIRS} #opencv
    ${G2O_INCLUDE_DIRS} #G2O
    ${CSPARSE_INCLUDE_DIR} #CSparse  稀疏矩阵求解
    "/usr/include/eigen3/" #Eigen3
)
# 特征提取
add_executable( feature_extraction feature_extraction.cpp  )
target_link_libraries( feature_extraction ${OpenCV_LIBS} )

# 2D-2D点对 求解 R t特征提取匹配 计算 本质举证E  基础矩阵E   单应矩阵 H   得到 旋转矩阵R 和 平移向量t
# add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp )
target_link_libraries( pose_estimation_2d2d ${OpenCV_LIBS} )

## 三角测量法得到深度
# add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2 
add_executable( triangulation triangulation.cpp )
target_link_libraries( triangulation ${OpenCV_LIBS} )

# 3D-2D点对 求解 R t 
# 2D-2D点对  其中一个 2D点通过 深度信息 转化成 3D点 和剩余的 一个 2D点组成 3D-2D点对 
# 透视法 PnP求解初始解 R t  通过 BA最小化重投影误差  优化 R  T 
add_executable( pose_estimation_3d2d pose_estimation_3d2d.cpp )
target_link_libraries( pose_estimation_3d2d 
   ${OpenCV_LIBS}
   ${CSPARSE_LIBRARY}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)

# 3D - 3D点对 求解 R t
# 1】 线性代数求解  SVD奇异值分解方法 两组 点集的质心位置 p  、 p‘  
# W= sum(qi * qi转置) = U * 对角矩阵 * V转置   奇异值分解      R= U * V转置  t =  p - R * p'
# 2】 非线性优化方法 ei = pi -exp(f) * pi'  = P - P‘   e 对 ∇f的导数  = P'对∇f的偏导数  数值优化 迭代更新
add_executable( pose_estimation_3d3d pose_estimation_3d3d.cpp )
target_link_libraries( pose_estimation_3d3d 
   ${OpenCV_LIBS}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension 
   ${CSPARSE_LIBRARY}
)
