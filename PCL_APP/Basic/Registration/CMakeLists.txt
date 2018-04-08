#编译器版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( pcl_Registration )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#找 pcl 并链接 
#注意　common io filters visualization　keypoints 分别为其子模块，
# 使用到子模块的需要添加相应的子模块
find_package( PCL REQUIRED COMPONENT common io registration visualization features filters)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )

# 使用ICP迭代最近点算法
add_executable (iterative_closest_point iterative_closest_point.cpp)
target_link_libraries (iterative_closest_point ${PCL_LIBRARIES})

# 使用迭代最近点算法，逐步实现地对一系列点云进行两两匹配
add_executable (pairwise_icp pairwise_incremental_registration.cpp)
target_link_libraries (pairwise_icp ${PCL_LIBRARIES})

# 交互式ICP可视化的程序
add_executable (interactive_icp interactive_icp.cpp)
target_link_libraries (interactive_icp  ${PCL_LIBRARIES})

# 使用正态分布变换进行配准的实验 
add_executable (ndt normal_distributions_transform.cpp)
target_link_libraries (ndt  ${PCL_LIBRARIES})

# 刚体的鲁棒位置估计 刚体　在场景点云中　匹配出位姿
add_executable (alignment_prerejective alignment_prerejective.cpp)
target_link_libraries (alignment_prerejective ${PCL_LIBRARIES})



