#编译器版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( pcl_filter )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#找 pcl 并链接 
find_package( PCL REQUIRED COMPONENT common io filters)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )

#可执行文件
add_executable( passthrough PassThroughfilter.cpp )

# 链接  PCL库
target_link_libraries( passthrough  ${PCL_LIBRARIES} )
