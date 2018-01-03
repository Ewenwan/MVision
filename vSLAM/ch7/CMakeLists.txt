#
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
    ${CSPARSE_INCLUDE_DIR} #CSparse
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
add_executable( pose_estimation_3d2d pose_estimation_3d2d.cpp )
target_link_libraries( pose_estimation_3d2d 
   ${OpenCV_LIBS}
   ${CSPARSE_LIBRARY}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)

# 3D - 3D点对 求解 R t
add_executable( pose_estimation_3d3d pose_estimation_3d3d.cpp )
target_link_libraries( pose_estimation_3d3d 
   ${OpenCV_LIBS}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension 
   ${CSPARSE_LIBRARY}
)
