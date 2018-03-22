#编译器版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( pcl_Tree_search )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#找 pcl 并链接 
#注意　common io filters visualization　keypoints 分别为其子模块，
# 使用到子模块的需要添加相应的子模块
find_package( PCL REQUIRED COMPONENT common visualization kdtree octree)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )

# kdtree Kd树按半径搜索或邻区搜索。
add_executable (kdtree_search kdTree_search.cpp)
target_link_libraries (kdtree_search ${PCL_LIBRARIES})

# octree 八叉树点云压缩
add_executable (octree_Compression octree_Point_Cloud_Compression.cpp)
target_link_libraries (octree_Compression ${PCL_LIBRARIES})

# octree 八叉树 空间划分及搜索操作（体素内近邻搜索 K近邻搜索 半径内近邻搜索）
add_executable (octree_search octree_search.cpp)
target_link_libraries ( octree_search ${PCL_LIBRARIES})

# octree 八叉树 无序点云数据集的空间变化检测
add_executable (octree_change_det octree_change_detection.cpp)
target_link_libraries ( octree_change_det ${PCL_LIBRARIES})


