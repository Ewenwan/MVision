#!/bin/sh
# 安装依赖项 参考 https://blog.csdn.net/vivian187/article/details/51747345?locationNum=12&fps=1
cd ..
# Project directory  工程目录
PTAMDir=`pwd`

# 系统更新
sudo apt-get update

# 安装基本的工具
# Base tools
sudo apt-get install libx11-dev libncurses5-dev libreadline6 libreadline-dev
# Build tools
sudo apt-get install build-essential cmake pkg-config
# Boost for C++ boost库
sudo apt-get install libboost-dev libboost-doc
# gfortran 
sudo apt-get install gfortran libgfortran-5-dev
# 3. 安装线性代数的低级库  Low level libraries for Linear Algebra 
# LAPCK&BLAS ---- 这是一个高效线性算术数学库，使用Fortran语言写的，主要用于数学矩阵运算
sudo apt-get install liblapack-dev libblas-dev libsuitesparse-dev
# 4. 图像IO 和 摄像机驱动Image I/O && Camera Driver
sudo apt-get install libjpeg-dev libpng-dev libtiff5-dev libdc1394-22-dev libv4l-dev
# 5. 视频IO， 编解码和 视频显示库Video I/O && Codec && Display
sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev libpostproc-dev libswscale-dev libavdevice-dev libsdl-dev
sudo apt-get install libgtk2.0-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
# 6. GUI OpenGL
sudo apt-get install mesa-common-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev
# qt4 & qt5
sudo apt-get install libqt4-dev qt5-default

# 第三方库
echo -e "\n Making 3rdParty directory in the current project directory..."
PTAM3rdParty=3rdParty
mkdir ${PTAM3rdParty}
cd ${PTAM3rdParty}
Path3rdParty=`pwd`

ulimit -s unlimited

# TooN (Tom’s Object-oriented numerics library) – 主要用于大量小矩阵的运算，尤其是矩阵分解和优化。
echo -e "\n Installing TooN... \n"
cd ${Path3rdParty}
# sudo git clone git://github.com/edrosten/TooN.git
wget https://www.edwardrosten.com/cvd/TooN-2.2.tar.xz
tar xvJf TooN-2.2.tar.xz
cd TooN-2.2
./configure && make && sudo make install

# libCVD (computer vision library) ---- 计算机视觉库，主要用于计算机视觉和视频、图片处理。
echo -e "\n Installing libCVD... \n"
cd ${Path3rdParty}
# sudo git clone git://github.com/edrosten/libcvd.git
wget https://www.edwardrosten.com/cvd/libcvd-20150407.tar.xz
tar xvJf libcvd-20150407.tar.xz
cd libcvd-20150407
./configure && make && sudo make install

# GVars3 (configuration system library) ---- 系统配置库，属于libCVD的子项目，功能是读取配置文件，获取命令行数
echo -e "\n Installing GVars3... \n"
cd ${Path3rdParty}
# sudo git clone git://github.com/edrosten/gvars.git
wget https://www.edwardrosten.com/cvd/gvars-3.0.tar.xz
tar xvJf gvars-3.0.tar.xz
cd gvars-3.0
./configure && make && sudo make install

echo -e "\n Make the libs work \n"
sudo ldconfig

echo -e "\n Install 3rdParties successfully! \n"

cd ${PTAMDir}

exit 0
