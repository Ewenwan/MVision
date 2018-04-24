# project
cmake_minimum_required (VERSION 2.8)
project(ELAS)
#编译模式 使用 IF(NOT ) ENDIF 放置重复设置
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

# 显示 编译模式信息
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

#LIST( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules/" )

#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
link_libraries(${OpenCV_LIBRARIES})

#包含库文件
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/src
)
# 自建库生成路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3 -std=c++0x")

# 创建共享库 SHARED 动态链接库
add_library(${PROJECT_NAME} SHARED
src/descriptor.cpp
src/elas.cpp
src/filter.cpp
src/matrix.cpp
src/triangle.cpp
)

add_executable( eLAS src/main.cpp )
target_link_libraries( eLAS ${OpenCV_LIBS} ${PROJECT_NAME} )

add_executable( video_eLAS src/main_Live.cpp )
target_link_libraries( video_eLAS ${OpenCV_LIBS} ${PROJECT_NAME} )
