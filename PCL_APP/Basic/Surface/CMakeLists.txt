#编译器版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( pcl_reconstruction )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#找 pcl 并链接 
#注意　common io filters visualization　keypoints 分别为其子模块，
# 使用到子模块的需要添加相应的子模块
find_package( PCL REQUIRED COMPONENT io common visualization surface kdtree filters segmentation)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )

# 滑动最小二乘 表面平滑　重建算法
add_executable (movingLSresampling resampling.cpp)
target_link_libraries (movingLSresampling  ${PCL_LIBRARIES})

# 把点云投影到平面上，在平面模型上提取凸（凹）多边形
add_executable (convex_hull2d convex_hull2d.cpp)
target_link_libraries (convex_hull2d  ${PCL_LIBRARIES})

# 无序点云的快速三角化  
# 得到点云文件的mesh网格文件ply 文件
# 使用贪婪投影三角化算法对有向点云进行三角化
add_executable (greedy_projection greedy_projection.cpp)
target_link_libraries (greedy_projection  ${PCL_LIBRARIES})

# 无序点云的B曲线拟合  >1.7　支持
#add_executable (bspline_fitting bspline_fitting.cpp)
#target_link_libraries (bspline_fitting  ${PCL_LIBRARIES})

# 三维重构之泊松重构
add_executable (recon_poisson recon_poisson.cpp)
target_link_libraries (recon_poisson  ${PCL_LIBRARIES})


# 三维重构之移动立方体算法
add_executable (marchingCubes recon_marchingCubes.cpp)
target_link_libraries (marchingCubes  ${PCL_LIBRARIES})




