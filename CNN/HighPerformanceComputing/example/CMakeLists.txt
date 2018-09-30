

#find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs )
#if(NOT OpenCV_FOUND)
#    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
#endif()

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )



include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

add_executable(squeezenet squeezenet.cpp)
target_link_libraries(squeezenet ncnn ${OpenCV_LIBS})

add_executable(mobilenet mobilenet.cpp)
target_link_libraries(mobilenet ncnn ${OpenCV_LIBS})

add_executable(fasterrcnn fasterrcnn.cpp)
target_link_libraries(fasterrcnn ncnn ${OpenCV_LIBS})

add_executable(yolov2 yolov2.cpp)
target_link_libraries(yolov2 ncnn ${OpenCV_LIBS})

# 在线检测模型 =====
add_executable(mobilenetv2ssdlite_camera mobilenetv2ssdlite_on_line.cpp)
target_link_libraries(mobilenetv2ssdlite_camera ncnn ${OpenCV_LIBS})

add_executable(mobilenetv2ssdlite mobilenetv2ssdlite.cpp)
target_link_libraries(mobilenetv2ssdlite ncnn ${OpenCV_LIBS})

add_executable(mobilenetv2ssdlite_coco mobilenetv2ssdlite-coco.cpp)
target_link_libraries(mobilenetv2ssdlite_coco ncnn ${OpenCV_LIBS})

add_executable(mobilenetssd mobilenetssd.cpp)
target_link_libraries(mobilenetssd ncnn ${OpenCV_LIBS})

add_executable(squeezenetssd squeezenetssd.cpp)
target_link_libraries(squeezenetssd ncnn ${OpenCV_LIBS})

