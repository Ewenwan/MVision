#版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( joinMap )

#模式
set( CMAKE_BUILD_TYPE Release )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen 
include_directories( "/usr/include/eigen3/" ) #实际安装位置

# pcl 
find_package( PCL REQUIRED COMPONENT common io )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

#可执行文件
add_executable( joinMap joinMap.cpp )

# 链接OpenCV库 PCL库
target_link_libraries( joinMap ${OpenCV_LIBS} ${PCL_LIBRARIES} )