#编译器版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( pcl_range_image )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#找 pcl 并链接 
#注意　common io filters visualization　keypoints 分别为其子模块，
# 使用到子模块的需要添加相应的子模块
find_package( PCL REQUIRED COMPONENT common io visualization range_image features)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )


# Range Images范围图像，意思是通过点云获得特定角度的观察图像.
add_executable( range_image_vis range_image_visualization.cpp)
target_link_libraries(range_image_vis ${PCL_LIBRARIES} )

# 生成深度图像（范围图像） 提取边界（根据深度信息映射到像素 再分割）
add_executable(ex_borders_f_range_img extract_borders_from_range_images.cpp)
target_link_libraries(ex_borders_f_range_img ${PCL_LIBRARIES} )


