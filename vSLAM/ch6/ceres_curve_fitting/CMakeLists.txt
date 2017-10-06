#版本限制
cmake_minimum_required( VERSION 2.8 )
#工程名
project( ceres_curve_fitting )
#模式
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用ceres库
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

# 包含OpenCV的头文件
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_executable( curve_fitting main.cpp )
# 添加与Ceres和OpenCV链接库(静态库　共享库)
target_link_libraries( curve_fitting ${CERES_LIBRARIES} ${OpenCV_LIBS} )