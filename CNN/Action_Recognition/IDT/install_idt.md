# 安装

## opencv2.4

##  ffmpeg
    sudo make install ffmpeg
    
## avformat
    sudo apt-get install libavformat-dev
## avdevice
    sudo apt-get install libavdevice-dev
    
## 安装idt

## 运行时找不到 opencv的一些动态链接库
    可查看可执行文件的链接情况 ：
        ldd 可执行文件
        可查看到哭文件的链接情况。
    在/etc/ld.so.conf.d目录中新建一个xxx.conf文件
    例如:
        sudo vim /etc/ld.so.conf.d/opencv.conf
    添加:
        /usr/local/lib
        /usr/local/lib/libopencv_highgui.so.2.4
    保存
    更新链接查找
    sudo ldconfig
