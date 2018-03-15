#编译器版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( pcl_filter )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#找 pcl 并链接 
#注意　common io filters visualization　分别为其子模块，
# 使用到子模块的需要添加相应的子模块
find_package( PCL REQUIRED COMPONENT common io filters visualization)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )

#可执行文件 直通滤波器 PassThrough　
add_executable( passthrough PassThroughfilter.cpp )
# 链接  PCL库
target_link_libraries( passthrough  ${PCL_LIBRARIES} )

# 体素格滤波器VoxelGrid 下采样
add_executable( VoxelGrid VoxelGrid_filter.cpp )
target_link_libraries( VoxelGrid  ${PCL_LIBRARIES} )

# 统计滤波器　率除去　离群点
add_executable( statisticalFilter statistical_removal_filter.cpp)
target_link_libraries( statisticalFilter  ${PCL_LIBRARIES} )

#  投影滤波　使用参数化模型投影点云　输出投影后的点的坐标
add_executable( project_filter project_inliers_filter.cpp)
target_link_libraries( project_filter  ${PCL_LIBRARIES} )

# 球半径滤波器
add_executable( radi_filter radius_outlier_filter.cpp)
target_link_libraries(radi_filter ${PCL_LIBRARIES} )

# 条件滤波器
add_executable( conditional_filter conditional_removal_filter.cpp)
target_link_libraries(conditional_filter ${PCL_LIBRARIES} )


