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
find_package( PCL REQUIRED COMPONENT common io filters visualization features)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )

# 简单可视化 CloudViewer 
add_executable( cloudViewer  cloud_viewer.cpp )
target_link_libraries( cloudViewer   ${PCL_LIBRARIES} )

# 功能最全的可视化类
add_executable( pCLVisualizer  pCLVisualizer.cpp )
target_link_libraries( pCLVisualizer   ${PCL_LIBRARIES} )

# 可视化深度图像
add_executable( range_image_visualization  range_image_visualization.cpp)
target_link_libraries(range_image_visualization   ${PCL_LIBRARIES} )


