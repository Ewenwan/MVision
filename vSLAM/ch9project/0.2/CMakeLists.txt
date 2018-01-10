#版本限制
cmake_minimum_required( VERSION 2.8 )
#工程名
project ( myslam )
#编译器
set( CMAKE_CXX_COMPILER "g++" )
#编译模式
set( CMAKE_BUILD_TYPE "Release" )
# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )

#包含 cmake_modules
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
#设置变量
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin ) #编译后的可执行 二进制文件
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib ) # 编译的库 存放地址

############### dependencies ######################
# Eigen
include_directories( "/usr/include/eigen3" )
# OpenCV
find_package( OpenCV 3.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# Sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )
# G2O
find_package( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    g2o_core g2o_stuff g2o_types_sba
)
############### source and test ######################
include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory( src )# 子目录 包含 CMakeList.txt文件
add_subdirectory( test )# 子目录 包含 CMakeList.txt文件

