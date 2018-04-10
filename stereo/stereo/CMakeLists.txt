#版本限制
cmake_minimum_required( VERSION 2.8 )

#工程名
project( Stereo_Calibr )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )

# pcl 
find_package( PCL REQUIRED COMPONENT common io )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

#双目相机采集程序
add_executable( Stereo_Cap stereo_cap.cpp )
# 链接OpenCV库
target_link_libraries( Stereo_Cap  ${OpenCV_LIBS} )

#双目相机矫正程序
add_executable( Stereo_Calibr stereo_calib.cpp )
# 链接OpenCV库
target_link_libraries( Stereo_Calibr  ${OpenCV_LIBS} )

#双目相机 立体匹配
add_executable( Stereo_Match stereo_match.cpp )
# 链接OpenCV库 pcl库
target_link_libraries( Stereo_Match ${OpenCV_LIBS} ${PCL_LIBRARIES})

#双目相机 立体匹配
add_executable( stereo_m_on_line stereo_match_on_line.cpp )
# 链接OpenCV库 pcl库
target_link_libraries(stereo_m_on_line ${OpenCV_LIBS})

# 双目匹配 + 生成点云
add_executable(stereo_pc stereo_match_point_cloud.cpp)
target_link_libraries(stereo_pc ${OpenCV_LIBS} ${PCL_LIBRARIES})
