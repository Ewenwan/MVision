
# DSO: Direct Sparse Odometry   直接法稀疏点云  SLAM
[代码](https://github.com/Ewenwan/dso)


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

# Pangolin  可视化库的使用

	参考地址：
	【1】Pangolin：https://github.com/stevenlovegrove/Pangolin
	【2】Pangolin安装问题：http://www.cnblogs.com/liufuqiang/p/5618335.html
	【3】Pangolin的Example：https://github.com/stevenlovegrove/Pangolin/tree/master/examples
	【4】Pangolin的使用：http://docs.ros.org/fuerte/api/pangolin_wrapper/html/namespacepangolin.html
	【5】特性：http://www.stevenlovegrove.com/?id=44

	https://www.cnblogs.com/shhu1993/p/6814714.html

