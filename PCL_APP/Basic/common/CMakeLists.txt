#编译器版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( pcl_point_matrix_transform )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#找 pcl 并链接 
#注意　common io filters visualization　keypoints 分别为其子模块，
# 使用到子模块的需要添加相应的子模块
find_package( PCL REQUIRED COMPONENT io common visualization)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )

# 点云坐标变换
add_executable (transform_Point transformPointC.cpp)
target_link_libraries (transform_Point ${PCL_LIBRARIES})

# 连接两个点云中的字段或数据形成新点云
add_executable (concatenate_cloud concatenate_clouds.cpp)
target_link_libraries (concatenate_cloud ${PCL_LIBRARIES})

# 获取深度传感器数据
add_executable (openni_grabber openni_grabber.cpp)
target_link_libraries (openni_grabber ${PCL_LIBRARIES})

# 计算 点云中心
add_executable (compute_cent compute_centroid.cpp)
target_link_libraries (compute_cent ${PCL_LIBRARIES})

# .pcd文件到.ply文件的转换
add_executable (pcd2ply pcd2ply.cpp)
target_link_libraries (pcd2ply ${PCL_LIBRARIES})





