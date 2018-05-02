# lad_slam 直接法 视觉里程计

# ros下安装
## 使用 老版本编译系统 rosmake
    sudo apt-get install python-rosinstall  
    mkdir ~/rosbuild_ws   #创建 编译空间
    cd ~/rosbuild_ws  
    rosws init . /opt/ros/indigo   #编译空间初始化
    mkdir package_dir  
    rosws set ~/rosbuild_ws/package_dir -t .  #设置
    echo "source ~/rosbuild_ws/setup.bash" >> ~/.bashrc  
    bash  
    cd package_dir 
## 2. 安装依赖包 
    sudo apt-get install ros-indigo-libg2o ros-indigo-cv-bridge liblapack-dev libblas-dev freeglut3-dev libqglviewer-dev libsuitesparse-dev libx11-dev  
## 3.下载包
    git clone https://github.com/tum-vision/lsd_slam.git lsd_slam  
## 4. 如果你需要openFabMap去闭环检测的话（可选） 需要opencv 非免费的包
    在 lsd_slam_core/CMakeLists.txt  
    中去掉下列四行注释即可
    #add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/openFabMap)  
    #include_directories(${PROJECT_SOURCE_DIR}/thirdparty/openFabMap/include)  
    #add_definitions("-DHAVE_FABMAP")  
    #set(FABMAP_LIB openFABMAP )  
    
    需要注意openFabMap需要OpenCv nonfree模块支持，OpenCV3.0以上已经不包含，作者推荐2.4.8版本
    其中nonfree模块可以由以下方式安装
    $ sudo add-apt-repository --yes ppa:xqms/opencv-nonfree  
    $ sudo apt-get update  
    $ sudo apt-get install libopencv-nonfree-dev  
    
## 5. 编译LSD-SLAM
    rosmake lsd_slam  
## 6. 运行LSD-SLAM
    1. 启动ROS服务               roscore  
    2. 启动摄像服务（USB摄像模式）  rosrun uvc_camera uvc_camera_node device:=/dev/video0  
    3. 启动LSD-viewer查看点云     rosrun lsd_slam_viewer viewer  
    4. 启动LSD-core  
       1）数据集模式
       rosrun lsd_slam_core dataset_slam _files:=<files> _hz:=<hz> _calib:=<calibration_file>  
       _files填数据集png包的路径，_hz表示帧率，填0代表默认值，_calib填标定文件地址 
       如：
       rosrun lsd_slam_core data
       set_slam _files:=<your path>/LSD_room/images _hz:=0 _calib:=<your path>/LSD_room/cameraCalibration.cfg  

       2）USB摄像模式
       rosrun lsd_slam_core live_slam /image:=<yourstreamtopic> _calib:=<calibration_file>  
       _calib同上，/image选择视频流 
       如：
       rosrun lsd_slam_core live_slam /image:=image_raw _calib:=<your path>/LSD_room/cameraCalibration.cfg 

## catkin_make编译
[catkin_make编译 参考](https://blog.csdn.net/zhuquan945/article/details/72980831)
  

### 编译错误记录 

#### opencv 
    KeyFrameDisplay.h
    
    //#include <opencv2/core/types_c.h>
    #include <opencv2/core.hpp>
    #include <opencv2/core/utility.hpp>
    
    
    
###  sophus 错误
    opt/ros/indigo/include/sophus/sim3.hpp:339:5: error: passing ‘const RxSO3Type {aka const Sophus::RxSO3Group}’ as ‘this’ argument of ‘void Sophus::RxSO3GroupBase::setScale(const Scalar&) [with Derived = Sophus::RxSO3Group; Sophus::RxSO3GroupBase::Scalar = double]’ discards qualifiers [-fpermissive]
    rxso3().setScale(scale);
    ^
    make[3]: *** [CMakeFiles/lsdslam.dir/src/DepthEstimation/DepthMap.cpp.o] Error 1
    make[3]: *** [CMakeFiles/lsdslam.dir/src/SlamSystem.cpp.o] Error 1


## 换用 catkin_make 编译
    1. 目录下有两个包 core 和 view 所以 在 lsd_slam目录下新建一个文件夹 lad_slam  
    把 CMakeLists.txt 和 package.xml 放入

    包信息  package.xml
           =========================
            <?xml version="1.0"?>
            <package>
              <name>lsd_slam</name>
              <version>0.0.0</version>
              <description>
                Large-Scale Direct Monocular SLAM
             </description>

              <author>Jakob Engel</author>
              <maintainer email="engelj@in.tum.de">Jakob Engel</maintainer>
              <license>see http://vision.in.tum.de/lsdslam </license>
              <url>http://vision.in.tum.de/lsdslam</url>

              <license>TODO</license>
              <buildtool_depend>catkin</buildtool_depend>
              <run_depend>lsd_slam_core</run_depend>
              <run_depend>lsd_slam_viewer</run_depend>

              <export>
                <metapackage/>
              </export>
            </package>
            ====================
    CMakeLists.txt    
            ====================
            cmake_minimum_required(VERSION 2.8.3)
            project(lsd_slam_robot)
            find_package(catkin REQUIRED)
            catkin_metapackage()
            ====================
### lsd_slam_viewer 
    包信息  package.xml
        ==============================
         <package>
          <name>lsd_slam_viewer_robot</name>
          <version>0.0.0</version>  

          <description>
             3D Viewer for LSD-SLAM.
          </description>

          <author>Jakob Engel</author>
          <maintainer email="engelj@in.tum.de">Jakob Engel</maintainer>
          <license>see http://vision.in.tum.de/lsdslam</license>
          <url>http://vision.in.tum.de/lsdslam</url>

          <buildtool_depend>catkin</buildtool_depend>
          <build_depend>cv_bridge</build_depend>
          <build_depend>dynamic_reconfigure</build_depend>
          <build_depend>sensor_msgs</build_depend>
          <build_depend>roscpp</build_depend>
          <build_depend>roslib</build_depend>
          <build_depend>rosbag</build_depend>
          <build_depend>message_generation</build_depend>
          <build_depend>cmake_modules</build_depend>

          <run_depend>cmake_modules</run_depend>
          <run_depend>cv_bridge</run_depend>
          <run_depend>dynamic_reconfigure</run_depend>
          <run_depend>sensor_msgs</run_depend>
          <run_depend>roscpp</run_depend>
          <run_depend>roslib</run_depend>
          <run_depend>rosbag</run_depend>

        </package>
       ======================================
     CMakeLists.txt 
       ======================================
        cmake_minimum_required(VERSION 2.4.6)
        project(lsd_slam_viewer_robot)

        #include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

        # Set the build type.  Options are:
        #  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
        #  Debug          : w/ debug symbols, w/o optimization
        #  Release        : w/o debug symbols, w/ optimization
        #  RelWithDebInfo : w/ debug symbols, w/ optimization
        #  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
        set(ROS_BUILD_TYPE Release)

        #rosbuild_init()

        ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/thirdparty/Sophus)

        set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
        set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

        find_package(catkin REQUIRED COMPONENTS
          cv_bridge
          dynamic_reconfigure
          sensor_msgs
          roscpp
          rosbag
          message_generation
          roslib
        )

        # 找opencv
        find_package( OpenCV REQUIRED )
        # 包含opencv
        #include_directories( ${OpenCV_INCLUDE_DIRS} )

        find_package(OpenGL REQUIRED)
        set(QT_USE_QTOPENGL TRUE)
        set(QT_USE_QTXML TRUE)
        find_package(QGLViewer REQUIRED)
        find_package(Eigen3 REQUIRED)

        #rosbuild_find_ros_package(dynamic_reconfigure)
        #include(${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake)
        #gencfg()

        find_package(Boost REQUIRED COMPONENTS thread)

        include_directories(${catkin_INCLUDE_DIRS}
                            ${QT_INCLUDES} 
                            ${EIGEN_INCLUDE_DIR} 
                            ${QGLVIEWER_INCLUDE_DIR}
                            ${OpenCV_INCLUDE_DIRS}
                            )

        # SSE flags
        set(CMAKE_CXX_FLAGS
           "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++0x"
        )

        # Messages & Services
        #rosbuild_genmsg()
        #消息文件
        add_message_files(DIRECTORY msg FILES keyframeMsg.msg keyframeGraphMsg.msg)
        generate_messages(DEPENDENCIES)
        #动态参数配置文件
        generate_dynamic_reconfigure_options(
          cfg/LSDSLAMViewerParams.cfg
        )

        # SSE flags
        rosbuild_check_for_sse()
        set(CMAKE_CXX_FLAGS
           "${SSE_FLAGS}"
        )

        # SSE Sources files
        set(SOURCE_FILES         
          src/PointCloudViewer.cpp
          src/KeyFrameDisplay.cpp
          src/KeyFrameGraphDisplay.cpp
          src/settings.cpp
          src/keyboard/keyboard.cc# keyboard
          src/robot/robot.cc# robor model
          src/serial/serial.cc# communication
        )

        set(HEADER_FILES     
          src/PointCloudViewer.h
          src/KeyFrameDisplay.h
          src/KeyFrameGraphDisplay.h
          src/settings.h
           src/keyboard/keyboard.h
           src/robot/robot.h
           src/serial/serial.h
        )

        include_directories(
          ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
        )  

        set(LIBS
        ${QGLViewer_LIBRARIES}
        ${QGLVIEWER_LIBRARY} 
        ${catkin_LIBRARIES}
        ${Boost_LIBRARIES}
        ${QT_LIBRARIES}
        ${OpenCV_LIBS}
        GL glut GLU X11
        )

        rosbuild_add_executable(viewer src/main_viewer.cpp ${SOURCE_FILES} ${HEADER_FILES})

        target_link_libraries(viewer ${LIBS})

        rosbuild_link_boost(viewer thread)

        rosbuild_add_executable(videoStitch src/main_stitchVideos.cpp)
        target_link_libraries(videoStitch ${OpenCV_LIBS})
      =============================
      
      
