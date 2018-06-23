# lad_slam 直接法 视觉里程计



#################
########################
lsd-slam  直接法稠密点云slam    Large Scale Direct Monocular
########################################
####################

	http://www.luohanjie.com/2017-03-17/ubuntu-install-lsd-slam.html
	https://vision.in.tum.de/research/vslam/lsdslam
	https://www.cnblogs.com/hitcm/p/4907536.html
	https://github.com/tum-vision/lsd_slam


官方编译方法[1]

	rosmake 编译
	sudo apt-get install python-rosinstall
	sudo apt-get install ros-indigo-libg2o ros-indigo-cv-bridge liblapack-dev libblas-dev freeglut3-dev libqglviewer-dev libsuitesparse-dev libx11-dev
	mkdir ~/SLAM/Code/rosbuild_ws
	cd ~/SLAM/Code/rosbuild_ws
	roses init . /opt/ros/indigo
	mkdir package_dir
	roses set ~/SLAM/Code/rosbuild_ws/package_dir -t .
	echo "source ~/SLAM/Code/rosbuild_ws/setup.bash" >> ~/.bashrc
	bash
	cd package_dir
	git clone https://github.com/tum-vision/lsd_slam.git lsd_slam
	rosmake lsd_slam


使用catkin对LSD-SLAM进行编译

	mkdir -p ~/catkin_ws/src
	git clone https://github.com/tum-vision/lsd_slam.git
	cd lsd_slam
	git checkout catkin

	对lsd_slam/lsd_slam_viewer和lsd_slam/lsd_slam_core文件夹下的package.xml中添加：
	<build_depend>cmake_modules</build_depend>
	<run_depend>cmake_modules</run_depend>

	对lsd_slam/lsd_slam_viewer和lsd_slam/lsd_slam_core文件夹下的CMakeFiles.txt中添加：
	find_package(cmake_modules REQUIRED)
	find_package(OpenCV 3.0 QUIET) #support opencv3
	if(NOT OpenCV_FOUND)
	   find_package(OpenCV 2.4.3 QUIET)
	   if(NOT OpenCV_FOUND)
	      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
	   endif()
	endif()


	并且在所有的target_link_libraries中添加X11 ${OpenCV_LIBS}，如：
	target_link_libraries(lsdslam 
	${FABMAP_LIB} 
	${G2O_LIBRARIES} 
	${catkin_LIBRARIES} 
	${OpenCV_LIBS} 
	sparse cxsparse X11
	)

然后开始编译：

	cd ~/catkin_ws/
	catkin_make


下载测试数据   474MB  日志回放
vmcremers8.informatik.tu-muenchen.de/lsd/LSD_room.bag.zip
解压

	打开一个终端:
	roscoe

	打开另外一个终端：
	cd ~/catkin_ws/
	source devel/setup.sh
	rosrun lsd_slam_viewer viewer

	打开另外一个终端：
	cd ~/catkin_ws/
	source devel/setup.sh
	rosrun lsd_slam_core live_slam image:=/image_raw camera_info:=/camera_info

	打开另外一个终端：
	cd ~/catkin_ws/
	rosbag play ~/LSD_room.bag     ###回放日志   即将之前的数据按话题发布


使用摄像头运行LSD_SLAM
安装驱动[4]：
	cd ~/catkin_ws/
	source devel/setup.sh
	cd ~/catkin_ws/src
	git clone https://github.com/ktossell/camera_umd.git
	cd ..
	catkin_make
	roscd uvc_camera/launch/
	roslaunch ./camera_node.launch

	camera_node.launch文件[5]，如：

	<launch>
	  <node pkg="uvc_camera" type="uvc_camera_node" name="uvc_camera" output="screen">
	    <param name="width" type="int" value="640" />
	    <param name="height" type="int" value="480" />
	    <param name="fps" type="int" value="30" />
	    <param name="frame" type="string" value="wide_stereo" />

	    <param name="auto_focus" type="bool" value="False" />
	    <param name="focus_absolute" type="int" value="0" />
	    <!-- other supported params: auto_exposure, exposure_absolute, brightness, power_line_frequency -->

	    <param name="device" type="string" value="/dev/video1" />
	    <param name="camera_info_url" type="string" value="file://$(find uvc_camera)/example.yaml" />
	  </node>
	</launch>

注意官方程序默认分辨率为640*480。

	打开一个窗口
	运行roscore；

	打开另外一个窗口：
	cd ~/catkin_ws/
	source devel/setup.sh
	rosrun lsd_slam_viewer viewer

	再打开另外一个窗口：
	cd ~/catkin_ws/
	source devel/setup.sh
	roslaunch uvc_camera camera_node.launch

	再打开另外一个窗口：
	rosrun lsd_slam_core live_slam /image:=image_raw _calib:=<calibration_file>
	校正文件calibration_file可参考lsd_catkin_ws/src/lsd_slam/lsd_slam_core/calib中的cfg文件。


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
            project(lsd_slam)
            find_package(catkin REQUIRED)
            catkin_metapackage()
            ====================
### lsd_slam_viewer 
    包信息  package.xml
        ==============================
        <package>
        <name>lsd_slam_viewer</name>
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
        project(lsd_slam_viewer)

        # Set the build type. Options are:
        #  Coverage : w/ debug symbols, w/o optimization, w/ code-coverage
        #  Debug : w/ debug symbols, w/o optimization
        #  Release : w/o debug symbols, w/ optimization
        #  RelWithDebInfo : w/ debug symbols, w/ optimization
        #  MinSizeRel : w/o debug symbols, w/ optimization, stripped binaries
        set(CMAKE_BUILD_TYPE Release)

        ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/thirdparty/Sophus)

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
        #find_package(OpenCV 2.4.3 REQUIRED)
        find_package(OpenCV 3.0 QUIET) #support opencv3
        if(NOT OpenCV_FOUND)
        find_package(OpenCV 2.4.3 QUIET)
        if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
        endif()
        endif()
        find_package(cmake_modules REQUIRED)

        find_package(OpenGL REQUIRED)
        set(QT_USE_QTOPENGL TRUE)
        set(QT_USE_QTXML TRUE)
        find_package(QGLViewer REQUIRED)
        find_package(Eigen REQUIRED)
        find_package(OpenCV REQUIRED)
        find_package(Boost REQUIRED COMPONENTS thread)

        include_directories(${QGLVIEWER_INCLUDE_DIR}
            ${catkin_INCLUDE_DIRS} 
            ${EIGEN_INCLUDE_DIR}
            ${QT_INCLUDES} )

        # SSE flags
        set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++0x"
        )

        add_message_files(DIRECTORY msg FILES keyframeMsg.msg keyframeGraphMsg.msg)
        generate_messages(DEPENDENCIES)

        generate_dynamic_reconfigure_options(
        cfg/LSDSLAMViewerParams.cfg
        )


        # Sources files
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
        GL glut GLU X11
        )

        add_executable(viewer src/main_viewer.cpp ${SOURCE_FILES} ${HEADER_FILES})
        target_link_libraries(viewer ${LIBS}
             )
        add_executable(videoStitch src/main_stitchVideos.cpp)
        target_link_libraries(videoStitch ${OpenCV_LIBS} ${LIBS})

        #add_executable(videoStitch src/main_stitchVideos.cpp)
        #target_link_libraries(viewer ${QGLViewer_LIBRARIES}
        #			     ${QGLVIEWER_LIBRARY}
        #			     ${catkin_LIBRARIES}
        #			     ${QT_LIBRARIES}
        #			     GL glut GLU
        #		      )

      =============================
      
 ### lsd_slam_core
 包信息  package.xml
         ========================
        <?xml version="1.0"?>
        <package>
          <name>lsd_slam_core</name>
          <version>0.0.0</version>
          <description>
             Large-Scale Direct Monocular SLAM
          </description>

          <author>Jakob Engel</author>
          <maintainer email="engelj@in.tum.de">Jakob Engel</maintainer>
          <license>see http://vision.in.tum.de/lsdslam </license>
          <url>http://vision.in.tum.de/lsdslam</url>

          <buildtool_depend>catkin</buildtool_depend>
          <build_depend>cv_bridge</build_depend>
          <build_depend>dynamic_reconfigure</build_depend>
          <build_depend>sensor_msgs</build_depend>
          <build_depend>roscpp</build_depend>
          <build_depend>lsd_slam_viewer</build_depend>
          <build_depend>rosbag</build_depend>
          <build_depend>eigen</build_depend>
          <build_depend>suitesparse</build_depend>
          <build_depend>libg2o</build_depend>
          <build_depend>cmake_modules</build_depend>

          <run_depend>cmake_modules</run_depend>
          <run_depend>cv_bridge</run_depend>
          <run_depend>dynamic_reconfigure</run_depend>
          <run_depend>sensor_msgs</run_depend>
          <run_depend>roscpp</run_depend>
          <run_depend>lsd_slam_viewer</run_depend>
          <run_depend>rosbag</run_depend>
          <run_depend>eigen</run_depend>
          <run_depend>suitesparse</run_depend>
          <run_depend>libg2o</run_depend>

        </package> 
        ===============
        
  CMakeLists.txt 
  
        ===============
        cmake_minimum_required(VERSION 2.8.7)
        project(lsd_slam_core)

        set(CMAKE_BUILD_TYPE Release)

        find_package(catkin REQUIRED COMPONENTS
          cv_bridge
          dynamic_reconfigure
          sensor_msgs
          image_transport
          roscpp
          rosbag
        )

        #find_package(OpenCV 2.4.3 REQUIRED)
        find_package(OpenCV 3.0 QUIET) #support opencv3
        if(NOT OpenCV_FOUND)
           find_package(OpenCV 2.4.3 QUIET)
           if(NOT OpenCV_FOUND)
              message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
           endif()
        endif()

        find_package(cmake_modules REQUIRED)

        find_package(Eigen3 REQUIRED)
        find_package(X11 REQUIRED)
        include(cmake/FindG2O.cmake)
        include(cmake/FindSuiteParse.cmake)

        message("-- CHOLMOD_INCLUDE_DIR : " ${CHOLMOD_INCLUDE_DIR})
        message("-- CSPARSE_INCLUDE_DIR : " ${CSPARSE_INCLUDE_DIR})
        message("-- G2O_INCLUDE_DIR : " ${G2O_INCLUDE_DIR})

        # FabMap
        # uncomment this part to enable fabmap
        #add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/openFabMap)
        #include_directories(${PROJECT_SOURCE_DIR}/thirdparty/openFabMap/include)
        #add_definitions("-DHAVE_FABMAP")
        #set(FABMAP_LIB openFABMAP )

        generate_dynamic_reconfigure_options(
          cfg/LSDDebugParams.cfg
          cfg/LSDParams.cfg
        )

        catkin_package(
          LIBRARIES lsdslam
          DEPENDS Eigen SuiteSparse
          CATKIN_DEPENDS libg2o 
        )

        # SSE flags
        add_definitions("-DUSE_ROS")
        add_definitions("-DENABLE_SSE")

        # Also add some useful compiler flag
        set(CMAKE_CXX_FLAGS
           "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++0x"
        ) 
        # Set source files
        set(lsd_SOURCE_FILES
          ${PROJECT_SOURCE_DIR}/src/DataStructures/Frame.cpp
          ${PROJECT_SOURCE_DIR}/src/DataStructures/FramePoseStruct.cpp
          ${PROJECT_SOURCE_DIR}/src/DataStructures/FrameMemory.cpp
          ${PROJECT_SOURCE_DIR}/src/SlamSystem.cpp
          ${PROJECT_SOURCE_DIR}/src/LiveSLAMWrapper.cpp
          ${PROJECT_SOURCE_DIR}/src/DepthEstimation/DepthMap.cpp
          ${PROJECT_SOURCE_DIR}/src/DepthEstimation/DepthMapPixelHypothesis.cpp
          ${PROJECT_SOURCE_DIR}/src/util/globalFuncs.cpp
          ${PROJECT_SOURCE_DIR}/src/util/SophusUtil.cpp
          ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
          ${PROJECT_SOURCE_DIR}/src/util/Undistorter.cpp
          ${PROJECT_SOURCE_DIR}/src/Tracking/Sim3Tracker.cpp
          ${PROJECT_SOURCE_DIR}/src/Tracking/Relocalizer.cpp
          ${PROJECT_SOURCE_DIR}/src/Tracking/SE3Tracker.cpp
          ${PROJECT_SOURCE_DIR}/src/Tracking/least_squares.cpp
          ${PROJECT_SOURCE_DIR}/src/Tracking/TrackingReference.cpp
          ${PROJECT_SOURCE_DIR}/src/IOWrapper/Timestamp.cpp
          ${PROJECT_SOURCE_DIR}/src/GlobalMapping/FabMap.cpp
          ${PROJECT_SOURCE_DIR}/src/GlobalMapping/KeyFrameGraph.cpp
          ${PROJECT_SOURCE_DIR}/src/GlobalMapping/g2oTypeSim3Sophus.cpp
          ${PROJECT_SOURCE_DIR}/src/GlobalMapping/TrackableKeyFrameSearch.cpp
        )
        set(SOURCE_FILES
          ${lsd_SOURCE_FILES}
          ${PROJECT_SOURCE_DIR}/src/IOWrapper/ROS/ROSImageStreamThread.cpp
          ${PROJECT_SOURCE_DIR}/src/IOWrapper/ROS/ROSOutput3DWrapper.cpp
          ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
        )
        include_directories(
          include
          ${EIGEN3_INCLUDE_DIR}
          ${PROJECT_SOURCE_DIR}/src
          ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
          ${CSPARSE_INCLUDE_DIR} #Has been set by SuiteParse
          ${CHOLMOD_INCLUDE_DIR} #Has been set by SuiteParse
        )
        set(LIBS
        ${catkin_LIBRARIES}
        ${G2O_LIBRARIES} 
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        )
        # build shared library.
        add_library(lsdslam SHARED ${SOURCE_FILES})
        target_link_libraries(lsdslam ${FABMAP_LIB} ${LIBS} csparse cxsparse 
        X11
        )
        #rosbuild_link_boost(lsdslam thread)
        # build live ros node
        add_executable(live_slam src/main_live_odometry.cpp)
        target_link_libraries(live_slam lsdslam ${LIBS}
        X11
        )
        # build image node
        add_executable(dataset src/main_on_images.cpp)
        target_link_libraries(dataset lsdslam ${LIBS}
        X11
        )
        # TODO add INSTALL

        ============
  
