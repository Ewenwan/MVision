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
find_package( PCL REQUIRED COMPONENT common io visualization features)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )


# 点云表面法线特征  协方差矩阵
add_executable( normal_3d_feature normal_3d_feature.cpp)
target_link_libraries( normal_3d_feature  ${PCL_LIBRARIES} )

# 积分图计算一个有序点云的法线，注意该方法只适用于有序点云
add_executable( integral_images_normal  normal_estimation_using_integral_images.cpp)
target_link_libraries( integral_images_normal ${PCL_LIBRARIES} )



