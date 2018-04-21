# 图像获取发布
cmake_minimum_required(VERSION 2.8)
project(image_publisher)

find_package(catkin REQUIRED COMPONENTS cv_bridge dynamic_reconfigure camera_info_manager sensor_msgs image_transport nodelet roscpp)

# 动态参数配置generate the dynamic_reconfigure config file
generate_dynamic_reconfigure_options(cfg/ImagePublisher.cfg)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

# 找opencv
find_package( OpenCV REQUIRED )
# 包含opencv
include_directories( ${OpenCV_INCLUDE_DIRS} )

#　发布图像类　源文件编译成　依赖库
add_library(${PROJECT_NAME} SHARED src/nodelet/image_publisher_nodelet.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS} )
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)
install(TARGETS image_publisher
        DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
#　实际节点
add_executable(image_publisher_exe src/node/image_publisher.cpp)
SET_TARGET_PROPERTIES(image_publisher_exe PROPERTIES OUTPUT_NAME image_publisher)#输出　 image_publisher　可执行文件
target_link_libraries(image_publisher_exe ${catkin_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS image_publisher_exe
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
# 安装插件
install(FILES nodelet_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
