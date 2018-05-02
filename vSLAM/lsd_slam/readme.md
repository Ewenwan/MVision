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
  
  
  
