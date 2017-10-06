cmake_minimum_required( VERSION 2.8 )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11" )

# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen 
include_directories( "/usr/include/eigen3/" )

# pcl 
find_package( PCL REQUIRED COMPONENT common io filters )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

# octomap 
find_package( octomap REQUIRED )
include_directories( ${OCTOMAP_INCLUDE_DIRS} )

add_executable( pointcloud_mapping pointcloud_mapping.cpp )
target_link_libraries( pointcloud_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES} )

add_executable( octomap_mapping octomap_mapping.cpp )
target_link_libraries( octomap_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES} )