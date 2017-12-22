
# 库介绍
* Pangolin是一个用于OpenGL显示/交互以及视频输入的一个轻量级、快速开发库
* Pangolin是对OpenGL进行封装的轻量级的OpenGL输入/输出和视频显示的库。
* 可以用于3D视觉和3D导航的视觉图，可以输入各种类型的视频、并且可以保留视频和输入数据用于debug。
* 下载工具Pangolin    github: https://github.com/stevenlovegrove/Pangolin

# 安装 pangolin 需要的依赖库 
OpenGL (Desktop / ES / ES2)依赖

Glew 依赖
sudo apt-get install libglew-dev     
CMake 依赖   编译需要
sudo apt-get install cmake
Boost 依赖    多线程  文件系统  是C++标准化进程的开发引擎之一
sudo apt-get install libboost-dev libboost-thread-dev libboost-filesystem-dev
python 依赖
sudo apt-get install libpython2.7-dev



#  编译 安装l pangolin
cd [path-to-pangolin]
mkdir build
cd build
cmake ..
make 
sudo make install 

# 编译此项目
* compile this program:
mkdir build
cd build
cmake ..
make 

* run the build/visualizeGeometry

2. How to use this program:

The UI in the left panel displays different representations of T_w_c ( camera to world ). 
显示 旋转矩阵 平移向量 欧拉角 四元素 
Drag your left mouse button to move the camera, 左键 移动相机
right button to rotate it around the box,                  右键 以箱子为中心旋转相机
center button to rotate the camera itself,                 中键 旋转相机本身
and press both left and right button to roll the view. 
Note that in this program the original X axis is right (red line), Y is up (green line) and Z in back axis (blue line). You (camera) are looking at (0,0,0) standing on (3,3,3) at first. 

3. Problems may happen:
* I found that in virtual machines there may be an error in pangolin, which was solved in its issue: https://github.com/stevenlovegrove/Pangolin/issues/74 . You need to comment the two lines mentioned by paulinus, and the recompile and reinstall Pangolin, if you happen to find this problem. 

If you still have problems using this program, please contact: gaoxiang12@mails.tsinghua.edu.cn
