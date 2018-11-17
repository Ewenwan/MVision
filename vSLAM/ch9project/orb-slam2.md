

# 1、ORBSLAM2
	ORBSLAM2在Ubuntu14.04上详细配置流程
	http://blog.csdn.net/zzlyw/article/details/54730830

## 1 安装必要工具
	首先，有两个工具是需要提前安装的。即cmake和git。
	sudo apt-get install cmake
	sudo apt-get install git

## 2 安装Pangolin，用于可视化和用户接口
	Pangolin： https://github.com/stevenlovegrove/Pangolin
	官方样例demo https://github.com/stevenlovegrove/Pangolin/tree/master/examples
	安装文件夹内
	Pangolin函数的使用：
	http://docs.ros.org/fuerte/api/pangolin_wrapper/html/namespacepangolin.html

	是一款开源的OPENGL显示库，可以用来视频显示、而且开发容易。
	是对OpenGL进行封装的轻量级的OpenGL输入/输出和视频显示的库。
	可以用于3D视觉和3D导航的视觉图，可以输入各种类型的视频、并且可以保留视频和输入数据用于debug。

	安装依赖项：
	http://www.cnblogs.com/liufuqiang/p/5618335.html  Pangolin安装问题
	Glew：   
	sudo apt-get install libglew-dev
	CMake：
	sudo apt-get install cmake
	Boost：
	sudo apt-get install libboost-dev libboost-thread-dev libboost-filesystem-dev
	Python2 / Python3：
	sudo apt-get install libpython2.7-dev
	sudo apt-get install build-essential

	先转到一个要存储Pangolin的路径下，例如~/Documents，然后
	git clone https://github.com/stevenlovegrove/Pangolin.git
	cd Pangolin
	mkdir build
	cd build
	cmake ..
	make -j
	sudo make install




## 3 安装OpenCV

	最低的OpenCV版本为2.4.3，建议采用OpenCV 2.4.11或者OpenCV 3.2.0。从OpenCV官网下载OpenCV2.4.11。然后安装依赖项：

	sudo apt-get install libgtk2.0-dev
	sudo apt-get install pkg-config

	将下载的OpenCV解压到自己的指定目录，然后cd到OpenCV的目录下。
	cd ~/Downloads/opencv-2.4.11
	mkdir release
	cd release
	cmake -D CMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local ..
	make
	sudo make install


## 4 安装Eigen3

	最低要求版本为3.1.0。在http://eigen.tuxfamily.org 下载Eigen3的最新版本，
	一般是一个压缩文件，下载后解压，然后cd到Eigen3的根目录下。

	mkdir build
	cd build
	cmake ..
	make
	sudo make install


## 5 安装ORBSLAM2

	先转到自己打算存储ORBSLAM2工程的路径，然后执行下列命令
	git clone https://github.com/raulmur/ORB_SLAM2.git oRB_SLAM2
	cd ORB_SLAM2
	修改编译 线程数(不然编译时可能会卡住)：
	vim build.sh
	最后 make -j >>>  make -j4

	sudo chmod 777 build.sh
	./build.sh


	之后会在lib文件夹下生成libORB_SLAM2.so，
	并且在Examples文件夹下生成
	mono_tum，mono_kitti， mono_euroc  in Examples/Monocular 单目 ，
	rgbd_tum   in Examples/Monocular RGB-D，
	stereo_kitti 和 stereo_euroc  in Examples/Stereo 双目立体。


## 数据集：
	KITTI dataset 对于 单目 stereo 或者 双目 monocular
	http://www.cvlibs.net/datasets/kitti/eval_odometry.php

	EuRoC dataset 对于 单目 stereo 或者 双目 monocular
	http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

	TUM dataset 对于 RGB-D 或者 单目monocular
	https://vision.in.tum.de/data/datasets/rgbd-dataset


## 论文：
ORB-SLAM: 
[Monocular] Raúl Mur-Artal, J. M. M. Montiel and Juan D. Tardós. ORB-SLAM: A Versatile and Accurate Monocular SLAM System. 
IEEE Transactions on Robotics, vol. 31, no. 5, pp. 1147-1163, 2015. (2015 IEEE Transactions on Robotics Best Paper Award). 
http://webdiis.unizar.es/%7Eraulmur/MurMontielTardosTRO15.pdf

ORB-SLAM2:
[Stereo and RGB-D] Raúl Mur-Artal and Juan D. Tardós. ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras. 
IEEE Transactions on Robotics, vol. 33, no. 5, pp. 1255-1262, 2017. 
https://128.84.21.199/pdf/1610.06475.pdf

词袋模型:
[DBoW2 Place Recognizer] Dorian Gálvez-López and Juan D. Tardós. Bags of Binary Words for Fast Place Recognition in Image Sequences. 
IEEE Transactions on Robotics, vol. 28, no. 5, pp. 1188-1197, 2012. 
http://doriangalvez.com/papers/GalvezTRO12.pdf


## 单目测试
	在http://vision.in.tum.de/data/datasets/rgbd-dataset/download下载一个序列，并解压。
	转到ORBSLAM2文件夹下，执行下面的命令。
	根据下载的视频序列freiburg1， freiburg2 和 freiburg3将TUMX.yaml分别转换为对应的 TUM1.yaml 或 TUM2.yaml 或 TUM3.yaml
	（相机参数文件）。
	将PATH_TO_SEQUENCE_FOLDER 更改为解压的视频序列文件夹。
	./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUMX.yaml PATH_TO_SEQUENCE_FOLDER 
											  解压的视频序列文件夹

## 双目测试
	在 http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets 下载一个序列 Vicon Room 1 02  大小1.2GB
	./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml PATH_TO_SEQUENCE/cam0/data PATH_TO_SEQUENCE/cam1/data Examples/Stereo/EuRoC_TimeStamps/SEQUENCE.txt


###################################
# 词带

 orb词带txt载入太慢，看到有人转换为binary，速度超快，试了下，确实快.
链接：https://github.com/raulmur/ORB_SLAM2/pull/21/commits/4122702ced85b20bd458d0e74624b9610c19f8cc     
Vocabulary/ORBvoc.txt >>> Vocabulary/ORBvoc.bin
################################################################
 
# CMakeLists.txt
	最后添加
	## .txt >>> .bin 文件转换
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools)
	add_executable(bin_vocabulary
	tools/bin_vocabulary.cc)
	target_link_libraries(bin_vocabulary ${PROJECT_NAME})

# build.sh   转换 .txt >>> .bin
	最后添加
	cd ..
	echo "Converting vocabulary to binary"
	./tools/bin_vocabulary

#### 新建转换文件
	tools/bin_vocabulary.cc

	#include <time.h>
	#include "ORBVocabulary.h"
	using namespace std;

	bool load_as_text(ORB_SLAM2::ORBVocabulary* voc, const std::string infile) {
	  clock_t tStart = clock();
	  bool res = voc->loadFromTextFile(infile);
	  printf("Loading fom text: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	  return res;
	}

	void load_as_xml(ORB_SLAM2::ORBVocabulary* voc, const std::string infile) {
	  clock_t tStart = clock();
	  voc->load(infile);
	  printf("Loading fom xml: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	}

	void load_as_binary(ORB_SLAM2::ORBVocabulary* voc, const std::string infile) {
	  clock_t tStart = clock();
	  voc->loadFromBinaryFile(infile);
	  printf("Loading fom binary: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	}

	void save_as_xml(ORB_SLAM2::ORBVocabulary* voc, const std::string outfile) {
	  clock_t tStart = clock();
	  voc->save(outfile);
	  printf("Saving as xml: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	}

	void save_as_text(ORB_SLAM2::ORBVocabulary* voc, const std::string outfile) {
	  clock_t tStart = clock();
	  voc->saveToTextFile(outfile);
	  printf("Saving as text: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	}

	void save_as_binary(ORB_SLAM2::ORBVocabulary* voc, const std::string outfile) {
	  clock_t tStart = clock();
	  voc->saveToBinaryFile(outfile);
	  printf("Saving as binary: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	}

	int main(int argc, char **argv) {
	  cout << "BoW load/save benchmark" << endl;
	  ORB_SLAM2::ORBVocabulary* voc = new ORB_SLAM2::ORBVocabulary();

	  load_as_text(voc, "Vocabulary/ORBvoc.txt");
	  save_as_binary(voc, "Vocabulary/ORBvoc.bin");

	  return 0;
	}

	修改读入文件：
	Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h
	line 248 
	添加
	// WYW ADD 2017.11.4 
	/**
	* Loads the vocabulary from a Binary file
	* @param filename
	*/
	bool loadFromBinaryFile(const std::string &filename);

	/**
	* Saves the vocabulary into a Binary file
	* @param filename
	*/
	void saveToBinaryFile(const std::string &filename) const;


	line 1460
	// WYW ADD 2017.11.4  读取二进制 词带
	// --------------------------------------------------------------------------
	template<class TDescriptor, class F>
	bool TemplatedVocabulary<TDescriptor,F>::loadFromBinaryFile(const std::string &filename) {
	fstream f;
	f.open(filename.c_str(), ios_base::in|ios::binary);
	unsigned int nb_nodes, size_node;
	f.read((char*)&nb_nodes, sizeof(nb_nodes));
	f.read((char*)&size_node, sizeof(size_node));
	f.read((char*)&m_k, sizeof(m_k));
	f.read((char*)&m_L, sizeof(m_L));
	f.read((char*)&m_scoring, sizeof(m_scoring));
	f.read((char*)&m_weighting, sizeof(m_weighting));
	createScoringObject();

	m_words.clear();
	m_words.reserve(pow((double)m_k, (double)m_L + 1));
	m_nodes.clear();
	m_nodes.resize(nb_nodes+1);
	m_nodes[0].id = 0;
	char buf[size_node]; int nid = 1;
	while (!f.eof()) {
	f.read(buf, size_node);
	m_nodes[nid].id = nid;
	// FIXME
	const int* ptr=(int*)buf;
	m_nodes[nid].parent = *ptr;
	//m_nodes[nid].parent = *(const int*)buf;
	m_nodes[m_nodes[nid].parent].children.push_back(nid);
	m_nodes[nid].descriptor = cv::Mat(1, F::L, CV_8U);
	memcpy(m_nodes[nid].descriptor.data, buf+4, F::L);
	m_nodes[nid].weight = *(float*)(buf+4+F::L);
	if (buf[8+F::L]) { // is leaf
	  int wid = m_words.size();
	  m_words.resize(wid+1);
	  m_nodes[nid].word_id = wid;
	  m_words[wid] = &m_nodes[nid];
	}
	else
	  m_nodes[nid].children.reserve(m_k);
	nid+=1;
	}
	f.close();
	return true;
	}

	// --------------------------------------------------------------------------
	template<class TDescriptor, class F>
	void TemplatedVocabulary<TDescriptor,F>::saveToBinaryFile(const std::string &filename) const {
	fstream f;
	f.open(filename.c_str(), ios_base::out|ios::binary);
	unsigned int nb_nodes = m_nodes.size();
	float _weight;
	unsigned int size_node = sizeof(m_nodes[0].parent) + F::L*sizeof(char) + sizeof(_weight) + sizeof(bool);
	f.write((char*)&nb_nodes, sizeof(nb_nodes));
	f.write((char*)&size_node, sizeof(size_node));
	f.write((char*)&m_k, sizeof(m_k));
	f.write((char*)&m_L, sizeof(m_L));
	f.write((char*)&m_scoring, sizeof(m_scoring));
	f.write((char*)&m_weighting, sizeof(m_weighting));
	for(size_t i=1; i<nb_nodes;i++) {
	const Node& node = m_nodes[i];
	f.write((char*)&node.parent, sizeof(node.parent));
	f.write((char*)node.descriptor.data, F::L);
	_weight = node.weight; f.write((char*)&_weight, sizeof(_weight));
	bool is_leaf = node.isLeaf(); f.write((char*)&is_leaf, sizeof(is_leaf)); // i put this one at the end for alignement....
	}
	f.close();
	}


	##### 修改slam系统文件   src/System.cc
	line 28
	// wyw添加 2017.11.4
	#include <time.h>
	bool has_suffix(const std::string &str, const std::string &suffix) {
	std::size_t index = str.find(suffix, str.size() - suffix.size());
	return (index != std::string::npos);
	}

	line 68
	/////// ////////////////////////////////////
	//// wyw 修改 2017.11.4
	clock_t tStart = clock();
	mpVocabulary = new ORBVocabulary();
	//bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
	bool bVocLoad = false; // chose loading method based on file extension
	if (has_suffix(strVocFile, ".txt"))
	  bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);//txt格式打开
	else
	  bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);//bin格式打开

	if(!bVocLoad)
	{
	cerr << "Wrong path to vocabulary. " << endl;
	cerr << "Failed to open at: " << strVocFile << endl;
	exit(-1);
	}
	//cout << "Vocabulary loaded!" << endl << endl;  
	printf("Vocabulary loaded in %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);//显示文件载入时间




	单目SLAM：
	例如，我自己的电脑上，该命令变为：
	./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/rgbd_dataset_freiburg1_xyz

	载入二进制词带
	./Examples/Monocular/mono_tum Vocabulary/ORBvoc.bin Examples/Monocular/TUM1.yaml /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/rgbd_dataset_freiburg1_xyz



	双目测试
	./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/ /cam0/data /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/ /cam1/data Examples/Stereo/EuRoC_TimeStamps/V102.txt
	载入二进制词带
	./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.bin Examples/Stereo/EuRoC.yaml /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/ /cam0/data /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/ /cam1/data Examples/Stereo/EuRoC_TimeStamps/V102.txt


	ros下的工程:
	http://blog.csdn.net/sinat_31802439/article/details/52331465  添加稠密地图
	https://pan.baidu.com/s/1miDA952


	manifest.xml >>>> package.xml

	<package>

	<name>ros_orb</name>     #####包名
	<version>0.0.1</version> #####版本
	<description>ORB_SLAM2</description>#####工程描述
	<author>EWenWan</author> ####作者
	<maintainer email="raulmur@unizar.es">Raul Mur-Artal</maintainer>##### 维护
	<license>GPLv3</license> ####开源协议

	<buildtool_depend>catkin</buildtool_depend> #### 编译工具以来

	<build_depend>roscpp</build_depend>         #### 编译依赖
	<build_depend>pcl</build_depend>
	<build_depend>tf</build_depend>
	<build_depend>sensor_msgs</build_depend>
	<build_depend>image_transport</build_depend>
	<build_depend>message_filters</build_depend>
	<build_depend>cv_bridge</build_depend>
	<build_depend>cmake_modules</build_depend>

	<run_depend>roscpp</run_depend>             #### 运行依赖
	<run_depend>pcl</run_depend>
	<run_depend>tf</run_depend>
	<run_depend>sensor_msgs</run_depend>
	<run_depend>image_transport</run_depend>
	<run_depend>message_filters</run_depend>
	<run_depend>cv_bridge</run_depend>

	</package>


	编译信息文件
	CMakeLists.txt

	cmake_minimum_required(VERSION 2.8.3) ### cmake版本限制

	project(ros_orb)##工程名
	find_package(catkin REQUIRED COMPONENTS###依赖包
	roscpp
	sensor_msgs
	image_transport
	message_filters
	cv_bridge
	cmake_modules)

	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

	### ORB_SLAM2的路径
	set(CODE_SOURCE_DIR /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/oRB_SLAM2/Examples/ROS/ORB_SLAM2)

	# Check C++11 or C++0x support
	include(CheckCXXCompilerFlag)
	CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
	CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
	if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	add_definitions(-DCOMPILEDWITHC11)
	message(STATUS "Using flag -std=c++11.")
	elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	add_definitions(-DCOMPILEDWITHC0X)
	message(STATUS "Using flag -std=c++0x.")
	else()
	message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
	endif()


	LIST(APPEND CMAKE_MODULE_PATH ${CODE_SOURCE_DIR}/../../../cmake_modules)## ORB_SLAM2的编译文件 FindEigen3.cmake

	find_package(OpenCV 2.4.3 REQUIRED)
	find_package(Eigen3 3.1.0 REQUIRED)
	find_package(Pangolin REQUIRED)
	find_package( G2O REQUIRED )
	find_package( PCL 1.7 REQUIRED )

	catkin_package()                      ###ros包类型说明 

	include_directories(
	${CODE_SOURCE_DIR}                    ### ORB_SLAM2的路径
	${CODE_SOURCE_DIR}/../../../
	${CODE_SOURCE_DIR}/../../../include
	${Pangolin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	)
	add_definitions( ${PCL_DEFINITIONS} )
	link_directories( ${PCL_LIBRARY_DIRS} )

	set(LIBS
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
	${EIGEN3_LIBS}
	${PCL_LIBRARIES}
	${Pangolin_LIBRARIES}
	${CODE_SOURCE_DIR}/../../../Thirdparty/DBoW2/lib/libDBoW2.so
	#g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sim3 g2o_types_sba
	${CODE_SOURCE_DIR}/../../../Thirdparty/g2o/lib/libg2o.so
	${CODE_SOURCE_DIR}/../../../lib/libORB_SLAM2.so
	)

	# Node for monocular camera 单目相机
	add_executable(mono
	src/ros_mono.cc
	)
	target_link_libraries(mono
	${LIBS}
	)
	# 单目相机 Augmented Reality 增强现实
	#add_executable(monoAR
	#src/AR/ros_mono_ar.cc
	#src/AR/ViewerAR.h
	#src/AR/ViewerAR.cc
	#)
	#target_link_libraries(mono
	#${LIBS}
	#)

	# Node for RGB-D camera 深度相机
	add_executable(rgbd
	src/ros_rgbd.cc
	)
	target_link_libraries(rgbd
	${LIBS}
	)

	# Node for stereo camera 双目立体相机
	add_executable(stereo
	src/ros_stereo.cc
	)
	target_link_libraries(stereo
	${LIBS}
	)

	cd catkin_ws
	catkin_make

	运行单目相机SLAM节点
	rosrun ros_orb mono Vocabulary/ORBvoc.bin Examples/Monocular/TUM1.yaml /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/rgbd_dataset_freiburg1_xyz




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



	###########################
	#################################
	#####################################
	DSO: Direct Sparse Odometry   直接法稀疏点云  SLAM  
	https://github.com/JakobEngel/dso


	１.下载DSO源代码到相应文件路径，比如我的文件路径为/home/hyj/DSO
	git clone https://github.com/JakobEngel/dso  dso
	２.安装suitesparse and eigen3 (必需)
	sudo apt-get install libsuitesparse-dev libeigen3-dev

	３.安装opencv. DSO对opencv依赖很少，仅仅用于读或写图像等一些简单的操作。
	sudo apt-get install libopencv-dev

	４.安装pangolin. 强烈推荐安装，考虑到ORB_SLAM中也选择pangolin作为显 示工具，而使用也非常方便，因此建议大家学习。 安装教程请移步pangolin的github主页

	５.安装ziplib. 建议安装，DSO用这个库来解压读取数据集压缩包中的图片，这样就不要每次都把下再的图片数据集进行解压了。
	sudo apt-get install zlib1g-dev
	cd thirdparty #找到DSO所在文件路径，切换到thirdparty文件夹下
	tar -zxvf libzip-1.1.1.tar.gz
	cd libzip-1.1.1/./configure
	make
	sudo make install
	sudo cp lib/zipconf.h /usr/local/include/zipconf.h

	6.编译DSO.
	cd /home/hyj/DSO/dso
	mkdir build
	cd build
	cmake ..
	make -j
	至此，不出意外的话，我们就可以很顺利的完成了DOS的安装。









	##############################
	###################################
	Pangolin  可视化库的使用
	参考地址：
	【1】Pangolin：https://github.com/stevenlovegrove/Pangolin
	【2】Pangolin安装问题：http://www.cnblogs.com/liufuqiang/p/5618335.html
	【3】Pangolin的Example：https://github.com/stevenlovegrove/Pangolin/tree/master/examples
	【4】Pangolin的使用：http://docs.ros.org/fuerte/api/pangolin_wrapper/html/namespacepangolin.html
	【5】特性：http://www.stevenlovegrove.com/?id=44

	https://www.cnblogs.com/shhu1993/p/6814714.html



