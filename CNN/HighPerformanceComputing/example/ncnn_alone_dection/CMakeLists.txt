cmake_minimum_required(VERSION 2.8.10)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

project(ncnn_alone)

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )

# 添加 ncnn头文件
include_directories(${CMAKE_SOURCE_DIR}/ncnn/include)
# 添加ncnn 链接库
#link_directories("./ncnn/lib")
#link_libraries(${CMAKE_SOURCE_DIR}/ncnn/lib/libncnn.so)


# 在线检测模型 自定义类=====
add_executable(mobilenetv2ssdlite_camera_my mobilenetv2ssdlite_on_line_my_class.cpp detector.cpp)
target_link_libraries(mobilenetv2ssdlite_camera_my ${CMAKE_SOURCE_DIR}/ncnn/lib/libncnn.a ${OpenCV_LIBS} -fopenmp)

