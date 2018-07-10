
# DSO: Direct Sparse Odometry   直接法稀疏点云  VO视觉里程计
[代码](https://github.com/Ewenwan/dso)

安装：
	
	DSO依赖项很少，并且这些依赖项也是大家很熟悉的库：Eigen3，pangolin，Opencv等

	１.下载DSO源代码到相应文件路径，比如我的文件路径为/home/hyj/DSO
	   git clone https://github.com/JakobEngel/dso  dso
	 
	 依赖项：
	２.安装suitesparse and eigen3 (必需)
	    sudo apt-get install libsuitesparse-dev libeigen3-dev
	    
	３.安装opencv. DSO对opencv依赖很少，仅仅用于读或写图像等一些简单的操作。
	    sudo apt-get install libopencv-dev

	４.安装pangolin. 强烈推荐安装，考虑到ORB_SLAM中也选择 pangolin 作为显示工具，而使用也非常方便，因此建议大家学习。 
	    安装教程请移步pangolin的github主页
	    https://github.com/Ewenwan/Pangolin

	５.安装ziplib. 建议安装，DSO用这个库来解压读取数据集压缩包中的图片，这样就不要每次都把下再的图片数据集进行解压了。
	    sudo apt-get install zlib1g-dev
	    
	    cd thirdparty # 找到DSO所在文件路径，切换到thirdparty文件夹下
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
	
# 在TUM单目数据集上运行DSO
	下载数据集 https://vision.in.tum.de/mono-dataset 
	cd /home/hyj/DSO/dso/build/bin 
	
		bin/dso_dataset \
		files=XXXXX/sequence_XX/images.zip \
		calib=XXXXX/sequence_XX/camera.txt \
		gamma=XXXXX/sequence_XX/pcalib.txt \
		vignette=XXXXX/sequence_XX/vignette.png \
		preset=0 \
		mode=0
		
	其中：
	files为数据集图片压缩包，
	calib为相机内参数文件(fx,fy,cx,cy, 畸变参数 k1 k2 r1 r2)，
	gamma和vignette为相机的一些特性参数，光度标定文件。
	mode为DSO模式切换，如0为包含光度表达文件，１为只包含内参数，２为没有畸变参数. 
	preset为设定DSO运行时的参数，如选取的像素点个数等等。
	preset＝3是preset＝0的5倍速运行DSO。
	preset＝0， 2000个点，无加速；
	preset＝1， 2000个点，1倍加速，；
	preset＝2， 800个点，无加速， 图像大小 424 x 320；
	preset＝3， 800个点，5倍加速，图像大小 424 x 320；
	
# 在自己单目数据集上运行DSO
	可能你平常实验中用自己的摄像头采集了一些数据，你手头只有图片，以及摄像头内参数，照样可以测试下DSO会表现如何。
	
	1. 准备参数文件
	   将自己内参数写入自己的camera.txt下，比如使用opencv或者matlab标定的针孔相机模型。
	   
		我的样例如下:
		446.867338 446.958766 298.082779 234.334299 -0.324849 0.1205156  -0.000186 -0.000821
		640 480 crop 640 480
		
		其中前八个数据就是我们熟悉的相机内参与畸变参数：fx fy cx cy k1 k2 r1 r2
		
		后面的参数为：
		输入图像尺寸 in_width in_height
		"crop" / "full" / "fx fy cx cy 0"
		输出图像尺寸 out_width out_height
		
	2. 准图片数据集
	   准备自己数据集的图片，注意图片名为６位，不足６位的补零，如下图所示。
	   c++可以通过setw(6)等指令来实现。
	   
	3.运行
           ./dso_dataset files=/home/hyj/bagfiles/img/ calib=/home/hyj/DSO/camera.txt mode=1
	
# 用自己摄像头实时运行DSO
	Engel同时发布了dsoros，用ROS来实时获取图片，程序代码很简短，
	实际上它是作者提供的一个如何把DSO当做一个黑盒子来使用的样例。
	根据dsoros的代码，你完全可以抛开ros，用opencv获取图片，然后去调用dso。
	
	




