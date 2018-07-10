# 可视化界面软件
    PCL Visialization,  Pangolin,  ros下的rviz
# 2. Pangolin 用于可视化和用户接口
## 安装，
	是一款开源的OPENGL显示库，可以用来视频显示、而且开发容易。
	是对OpenGL进行封装的轻量级的OpenGL输入/输出和视频显示的库。
	可以用于3D视觉和3D导航的视觉图，
	可以输入各种类型的视频、并且可以保留视频和输入数据用于debug。

[github](https://github.com/Ewenwan/Pangolin)

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
