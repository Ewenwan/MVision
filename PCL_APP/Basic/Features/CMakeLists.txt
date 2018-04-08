#编译器版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( pcl_feature )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#找 pcl 并链接 
#注意　common io filters visualization　分别为其子模块，
# 使用到子模块的需要添加相应的子模块
find_package( PCL REQUIRED COMPONENT common io visualization features  keypoints )
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )


# 点云表面法线特征  协方差矩阵
add_executable( normal_3d_feature normal_3d_feature.cpp)
target_link_libraries( normal_3d_feature  ${PCL_LIBRARIES} )

# 积分图计算一个有序点云的法线，注意该方法只适用于有序点云
add_executable( integral_images_normal  normal_estimation_using_integral_images.cpp)
target_link_libraries( integral_images_normal ${PCL_LIBRARIES} )

# 点云 phf点特征直方图 n×k^2(k个点之间)特征 计算法线 临近点对法线夹角（r p y） 直方图
# 临近k个点之间 k×k个点对 法线角度差(r p y) + 距离
add_executable( pFH_features PFH_features.cpp)
target_link_libraries( pFH_features  ${PCL_LIBRARIES} )

# 点云 fphf 快速点特征直方图 n×k(查询点和k个临近点间)
# 当前点 和周围k个点之间 k对点对 法线角度差(r p y) + 距离
add_executable( fpFH_features fasterPFH_features.cpp)
target_link_libraries( fpFH_features  ${PCL_LIBRARIES})

# 点云 vhf 视角点云特征 = 视角特征 + 快速点特征直方图 
add_executable( vFH_Feature vFH_Feature.cpp)
target_link_libraries( vFH_Feature  ${PCL_LIBRARIES})

# 点云  narf特征 深度图像(RangeImage) 边缘点 
add_executable( narf_feature narf_feature.cpp)
target_link_libraries( narf_feature  ${PCL_LIBRARIES})

# 点云包围盒 特征 估计
add_executable( bounding_boxes bounding_boxes_fea.cpp)
target_link_libraries( bounding_boxes  ${PCL_LIBRARIES})

# RoPs特征(Rotational Projection Statistics) 描述子
# 领域视角旋转投影 盒子 (熵 中心矩)述
# 旋转投影统计特征 
add_executable( rOPS_feature rOPSEstimation.cpp)
target_link_libraries( rOPS_feature  ${PCL_LIBRARIES})

# 全局一致的空间分布描述子特征 >1.9 才有
#add_executable(  gASD_feature Globally_Aligned_Spatial_Distribution.cpp)
#target_link_libraries( gASD_feature  ${PCL_LIBRARIES})

