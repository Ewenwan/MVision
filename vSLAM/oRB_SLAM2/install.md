# ORB_slam2 安装 词带转换修改　运行测试
[ORBSLAM2在Ubuntu14.04上详细配置流程](http://blog.csdn.net/zzlyw/article/details/54730830)

# 要求

操作系统 Ubuntu 12.04、14.04、16.04 

C++11 or C++0x 编译环境 使用最新的线程 thread 和 计时chrono 函数功能

[可视化 和人机接口 Pangolin](https://github.com/stevenlovegrove/Pangolin) 

[Pangolin　安装方法]https://github.com/stevenlovegrove/Pangolin.

OpenCV 计算机视觉库
[OpenCV](http://opencv.org)　大于2.4.3　 　2.4.11 　3.2　

[Eigen3 矩阵运算库](http://eigen.tuxfamily.org)  大于 3.1.0　

[词带模型DBoW2](https://github.com/dorian3d/DBoW2)

[图优化g2o](https://github.com/RainerKuemmerle/g2o) 

[ROS (optional) 机器人系统](ros.org)


## 一、安装依赖工具
### 1 安装必要工具
	首先，有两个工具是需要提前安装的。即cmake和git。

	sudo apt-get install cmake
	sudo apt-get install git

### 2 安装Pangolin，用于可视化和用户接口
	是一款开源的OPENGL显示库，可以用来视频显示、而且开发容易。
	是对OpenGL进行封装的轻量级的OpenGL输入/输出和视频显示的库。
	可以用于3D视觉和3D导航的视觉图，
	可以输入各种类型的视频、并且可以保留视频和输入数据用于debug。

[github](https://github.com/stevenlovegrove/Pangolin)

[官方样例demo](https://github.com/stevenlovegrove/Pangolin/tree/master/examples)

[Pangolin函数的使用](http://docs.ros.org/fuerte/api/pangolin_wrapper/html/namespacepangolin.html)


[Pangolin安装问题 安装依赖项](http://www.cnblogs.com/liufuqiang/p/5618335.html)

> 安装Pangolin的依赖项

	1. glew
	   sudo apt-get install libglew-dev
	2. CMake：
	   sudo apt-get install cmake
	3. Boost：
	   sudo apt-get install libboost-dev libboost-thread-dev libboost-filesystem-dev
	4. Python2 / Python3：
	   sudo apt-get install libpython2.7-dev
	5. build-essential
	   sudo apt-get install build-essential
> 安装Pangolin

	git clone https://github.com/stevenlovegrove/Pangolin.git
	cd Pangolin
	mkdir build
	cd build
	cmake ..
	make -j
	sudo make install

### 3. 安装OpenCV

	最低的OpenCV版本为2.4.3，
	建议采用OpenCV 2.4.11或者OpenCV 3.2.0。
	从OpenCV官网下载OpenCV2.4.11。
	然后安装依赖项：
		sudo apt-get install libgtk2.0-dev
		sudo apt-get install pkg-config

	将下载的OpenCV解压到自己的指定目录，然后cd到OpenCV的目录下。
		cd ~/opencv-2.4.11
		mkdir release
		cd release
		cmake -D CMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local ..
		make
		sudo make install


### 4. 安装Eigen3
	最低要求版本为3.1.0。

[Eigen3的最新版本](http://eigen.tuxfamily.org)

	cd Eigen3
	mkdir build
	cd build
	cmake ..
	make
	sudo make install


## 二、安装ORBSLAM2

> 先转到自己打算存储ORBSLAM2工程的路径，然后执行下列命令：

        下载：
		git clone https://github.com/raulmur/ORB_SLAM2.git oRB_SLAM2

        编译：
		cd ORB_SLAM2
		修改编译 线程数(不然编译时可能会卡住)：
		vim build.sh
		最后 make -j >>>  make -j4
　　　　执行编译脚本：
		sudo chmod 777 build.sh
		./build.sh


> 之后会在lib文件夹下生成libORB_SLAM2.so，并且在Examples文件夹下生成

	mono_tum，mono_kitti， mono_euroc  in Examples/Monocular 单目 ，
	rgbd_tum   in Examples/Monocular RGB-D，
	stereo_kitti 和 stereo_euroc  in Examples/Stereo 双目立体。


## 三、数据集：

[KITTI dataset 对于 单目 stereo 或者 双目 monocular](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

[EuRoC dataset 对于 单目 stereo 或者 双目 monocular](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

[TUM dataset 对于 RGB-D 或者 单目monocular](https://vision.in.tum.de/data/datasets/rgbd-dataset)


# 四、相关论文：

[ORB-SLAM 单目Monocular特征点法](http://webdiis.unizar.es/%7Eraulmur/MurMontielTardosTRO15.pdf)

[ORB-SLAM2 单目双目rgbd](https://128.84.21.199/pdf/1610.06475.pdf)

[词袋模型DBoW2 Place Recognizer](http://doriangalvez.com/papers/GalvezTRO12.pdf)


# 五、单目测试

[TUM Dataset数据集 下载一个单目测试集，并解压](http://vision.in.tum.de/data/datasets/rgbd-dataset/download)
	
	转到ORBSLAM2文件夹下，执行下面的命令。
	根据下载的视频序列freiburg1， freiburg2 和 freiburg3将
	TUMX.yaml分别转换为对应的 TUM1.yaml 或 TUM2.yaml 或 TUM3.yaml（相机参数文件）。
	将PATH_TO_SEQUENCE_FOLDER 更改为解压的视频序列文件夹。
　　　　 可执行文件 地图特征字典 数据集使用得相机参数　数据集地址
```sh
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUMX.yaml PATH_TO_SEQUENCE_FOLDER 
```

[KITTI Dataset 数据集](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
	不同的数据集序列对应不同得相机参数 
	sequence 0 to 2, 3, and 4 to 12 对应 KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml
	对有参数修改为 KITTIX.yaml　
	数据集路径 `PATH_TO_DATASET_FOLDER`
	时间序列 `SEQUENCE_NUMBER` to 00, 01, 02,.., 11. 
```sh
./Examples/Monocular/mono_kitti Vocabulary/ORBvoc.txt Examples/Monocular/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```

[EuRoC Dataset数据集 ](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

```
./Examples/Monocular/mono_euroc Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml PATH_TO_SEQUENCE_FOLDER/mav0/cam0/data Examples/Monocular/EuRoC_TimeStamps/SEQUENCE.txt 
```
```
./Examples/Monocular/mono_euroc Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml PATH_TO_SEQUENCE/cam0/data Examples/Monocular/EuRoC_TimeStamps/SEQUENCE.txt 
```


# 六、双目测试
[EuRoC Dataset 数据集 下载一个序列 Vicon Room 1 02  大小1.2GB](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
```sh
./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml PATH_TO_SEQUENCE/cam0/data 
 Examples/Stereo/EuRoC_TimeStamps/SEQUENCE.txt
```
```
./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml PATH_TO_SEQUENCE/cam0/data PATH_TO_SEQUENCE/cam1/data Examples/Stereo/EuRoC_TimeStamps/SEQUENCE.txt
```

[KITTI Dataset数据集](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) 

	不同的数据集序列对应不同得相机参数 
	sequence 0 to 2, 3, and 4 to 12 对应 KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml
	对有参数修改为 KITTIX.yaml　
	数据集路径 `PATH_TO_DATASET_FOLDER`
	时间序列 `SEQUENCE_NUMBER` to 00, 01, 02,.., 11. 
```
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```

# 七. RGB-D 深度相机示例　

[TUM Dataset数据集](http://vision.in.tum.de/data/datasets/rgbd-dataset/download)

>**匹配RGB图像和深度图像

[associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools)

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```
  
        根据下载的视频序列freiburg1， freiburg2 和 freiburg3将
	TUMX.yaml分别转换为对应的 TUM1.yaml 或 TUM2.yaml 或 TUM3.yaml（相机参数文件）。
	将PATH_TO_SEQUENCE_FOLDER 更改为解压的视频序列文件夹。
	ASSOCIATIONS_FILE 匹配文件
　　　 可执行文件 地图特征字典 数据集使用得相机参数　数据集地址 匹配文件
```
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```

# 八、词带格式转换
	 orb词带txt载入太慢，
	 看到有人转换为binary，
	 速度超快，试了下，确实快.
 
[参考链接](https://github.com/raulmur/ORB_SLAM2/pull/21/commits/4122702ced85b20bd458d0e74624b9610c19f8cc)

	Vocabulary/ORBvoc.txt >>> Vocabulary/ORBvoc.bin

## 1. 修改CMakeLists.txt文件　添加转换工具
	#CMakeLists.txt
	最后添加
	## .txt >>> .bin 文件转换
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools)
	add_executable(bin_vocabulary
	tools/bin_vocabulary.cc)
	target_link_libraries(bin_vocabulary ${PROJECT_NAME})
	
## 2. 修改 build.sh脚本　将　转换 .txt >>> .bin
	# build.sh   
	最后添加
	cd ..
	echo "Converting vocabulary to binary"
	./tools/bin_vocabulary

## 3. 新建转换文件 tools/bin_vocabulary.cc
```c
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
```

## 4. 修改orb系统读入字典的接口 TemplatedVocabulary.h

	Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h
	
> 在行 248　处左右 

	添加
	// WYW ADD 2017.11.4 
	  /**
	   * Loads the vocabulary from a Binary file
	   * @param filename  读取　函数声明
	   */
	  bool loadFromBinaryFile(const std::string &filename);

	  /**
	   * Saves the vocabulary into a Binary file
	   * @param filename  保存
	   */
	  void saveToBinaryFile(const std::string &filename) const;

> 在行 1460处左右
```c
	// WYW ADD 2017.11.4  读取二进制 词带 的函数
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

	// ----------------保存二进制词带文件----------------------------------------
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
```

## 5. 修改slam系统文件   src/System.cc
> 在行 28处左右 
```c
	// wyw添加 2017.11.4
	#include <time.h>
	// 文件类型判断
	bool has_suffix(const std::string &str, const std::string &suffix) {
	  std::size_t index = str.find(suffix, str.size() - suffix.size());
	  return (index != std::string::npos);
	}
```
> 在行  68处左右
```c
	/////// ////////////////////////////////////
	//// wyw 修改 2017.11.4
	    clock_t tStart = clock();
	    mpVocabulary = new ORBVocabulary();
	    //bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);// txt格式　词典
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
```
## 6. 进行测试
>**单目SLAM**

	例如，我自己的电脑上，该命令变为：
	./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/rgbd_dataset_freiburg1_xyz

	载入二进制词带
	./Examples/Monocular/mono_tum Vocabulary/ORBvoc.bin Examples/Monocular/TUM1.yaml /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/rgbd_dataset_freiburg1_xyz


>**双目测试**

	./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/ /cam0/data /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/ /cam1/data Examples/Stereo/EuRoC_TimeStamps/V102.txt
	载入二进制词带
	./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.bin Examples/Stereo/EuRoC.yaml /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/ /cam0/data /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/ /cam1/data Examples/Stereo/EuRoC_TimeStamps/V102.txt


# 九、ros下的工程:

[添加稠密地图](http://blog.csdn.net/sinat_31802439/article/details/52331465)

[修改后的工程](https://pan.baidu.com/s/1miDA952)

> **添加环境变量
```
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH/ORB_SLAM2/Examples/ROS
```
> **执行编译脚本

  ```
  chmod +x build_ros.sh
  ./build_ros.sh
  ```
  
>**单目相机节点

	单目相机话题　`/camera/image_raw` 
	节点
	rosrun ORB_SLAM2 Mono PATH_TO_VOCABULARY PATH_TO_SETTINGS_FILE

> **单目AR 增强现实 Augmented Reality Demo
　　　　
	单目相机话题　`/camera/image_raw` 
	节点
	  rosrun ORB_SLAM2 MonoAR PATH_TO_VOCABULARY PATH_TO_SETTINGS_FILE

> **双目相机节点

	双目相机话题
	`/camera/image_raw` 
	`/camera/left/image_raw` 
	`/camera/right/image_raw`
	节点
	```
	rosrun ORB_SLAM2 Stereo PATH_TO_VOCABULARY PATH_TO_SETTINGS_FILE ONLINE_RECTIFICATION
	```
  
	```
	roscore
	```

	```
	rosrun ORB_SLAM2 Stereo Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml true
	```

	```
	rosbag play --pause V1_01_easy.bag /cam0/image_raw:=/camera/left/image_raw /cam1/image_raw:=/camera/right/image_raw
	```
  

> **RGB_D相机

	相机话题
	`/camera/rgb/image_raw` 
	`/camera/depth_registered/image_raw`
	节点
	  ```
	  rosrun ORB_SLAM2 RGBD PATH_TO_VOCABULARY PATH_TO_SETTINGS_FILE
	  ```
## 可能要修改的地方
### 1. manifest.xml >>>> package.xml

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


### 2. 编译信息文件 CMakeLists.txt

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
	
### 3. 编译并运行
	cd catkin_ws
	catkin_make

	运行单目相机SLAM节点
	rosrun ros_orb mono 
	　Vocabulary/ORBvoc.bin 
	　Examples/Monocular/TUM1.yaml 
	 /home/ewenwan/ewenwan/learn/vSLAM/test/vSLAM/ch9project/date/rgbd_dataset_freiburg1_xyz
