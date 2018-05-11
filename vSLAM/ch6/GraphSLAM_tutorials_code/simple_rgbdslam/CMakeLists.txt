#版本要求
cmake_minimum_required(VERSION 2.8)
#工程名
project(rgbdslam)
#
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#设置编译 lib输出文件夹
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# 社祝编制后的执行文件 输出文件夹
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#添加子库  包含CMakeList.txt文件
ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src )
