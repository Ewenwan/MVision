#编译器版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( segmentation )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#找 pcl 并链接 
#注意　common io filters visualization　keypoints 分别为其子模块，
# 使用到子模块的需要添加相应的子模块
find_package( PCL REQUIRED COMPONENT io common visualization filters segmentation filters)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )

# 1 随机采样一致性 球模型 和 平面模型
add_executable (random_sample_consensus random_sample_consensus.cpp)
target_link_libraries (random_sample_consensus ${PCL_LIBRARIES})

# 2 分割平面　
add_executable (planar_segmentation planar_segmentation.cpp)
target_link_libraries (planar_segmentation ${PCL_LIBRARIES})

# 3 圆柱体分割 平面模型分割平面　平面上按　圆柱体模型分割得到圆柱体点云
add_executable (cylinder_segmentation cylinder_segmentation.cpp)
target_link_libraries (cylinder_segmentation ${PCL_LIBRARIES})

# 4 欧式聚类分割　平面模型分割平面　平面上按　聚类得到　多个点云团
add_executable (clusters_segmentation clusters_segmentation.cpp)
target_link_libraries (clusters_segmentation ${PCL_LIBRARIES})

# 5 基于　法线差值　和　曲率差值的　区域聚类分割算法
add_executable (region_growing_normal_cur region_growing_normal_cur.cpp)
target_link_libraries (region_growing_normal_cur ${PCL_LIBRARIES})

# 6 基于颜色的　区域聚类分割算法
add_executable (color_based color_based_region_growing_segmentation.cpp)
target_link_libraries (color_based ${PCL_LIBRARIES})

# 7 最小分割算法  (分割点云) 基于距离加权的最小图分割算法
add_executable (min_Cut min_Cut_Based_Segmentation.cpp)
target_link_libraries (min_Cut ${PCL_LIBRARIES})

#  8 基于不同领域半径估计的　法线的差异类分割点云
add_executable (difference_of_Normals Difference_of_Normals_in_diff_radis__Segmentation.cpp)
target_link_libraries (difference_of_Normals ${PCL_LIBRARIES})

# 9 超体聚类是一种图像的分割方法
add_executable (supervoxel supervoxel_clustering.cpp)
target_link_libraries (supervoxel ${PCL_LIBRARIES})

# 10 使用渐进式形态学滤波器 识别地面 
add_executable (morphologicalFilter_seg ProgressiveMorphologicalFilter_segmentation.cpp)
target_link_libraries (morphologicalFilter_seg ${PCL_LIBRARIES})

