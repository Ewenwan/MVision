#编译器版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( pcl_keypoints_extract )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#找 pcl 并链接 
#注意　common io filters visualization　keypoints 分别为其子模块，
# 使用到子模块的需要添加相应的子模块
find_package( PCL REQUIRED COMPONENT common io filters visualization keypoints)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )


# NARF　关键点　点云对应深度图　边缘检测　之上提取关键点
add_executable( narf_kp_extr narf_keypoint_extraction.cpp)
target_link_libraries(narf_kp_extr ${PCL_LIBRARIES} )

# harris3d 关键点
add_executable(harris3d harris3d_keypoint_extraction.cpp)
target_link_libraries(harris3d ${PCL_LIBRARIES} )


