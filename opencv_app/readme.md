# opencv 学习

## window下安装
      1、系统环境变量设置
      动态链接库配置
      计算机 -> 右键属性 ->高级系统设置 -> 高级标签  -> 最下边 环境变量

      ->在系统变量 path 中添加 路径 在原有路径后添加冒号;

      -> x64 位添加两个 路径 ...opencv\build\x86\vc11\bin;...opencv\build\x64\vc11\bin;
      (vc8 对应vc2008 vc9对应vc2009  vc10对应vc2010   vc11对应vc2012的版本  vc12 对应2013 
      vc13：vs2014 
      vc14：vs2015   vc15:vs2017 )

      -> x86  添加一个 路径 ...opencv\build\x86\vc11\bin;

      2、vs工程包含目录

      文件 -> 新建 -> 项目  -> win32控制台应用程序
      -> 进入win32应用程序设置向导
      -> 附加选项空项目  项目名
      -> 在解决方案资源管理器的源文件下
      -> 右键单击 添加 新建项  添加c++文件


      -> 打开属性管理器 （视图 属性管理器）
      -> 在属性管理器里配置一次 相当于进行了通用配置
      -> 打开属性管理器 工作区
      -> 展开Debug|Win32
      -> 对 Microsoft.Cpp.Win32.userDirectories 右键 属性配置

      -> 首先在 通用属性 
      -> VC++目录
      -> 包含目录 下添加三个路径
      ..\opencv\build\include
      ..\opencv\build\include\opencv
      ..\opencv\build\include\opencv2

      -> 首先在 通用属性 
      -> VC++目录
      -> 库目录 下添加  一个路径
      ..\opencv\build\x64\vc14\lib


      -> 首先在 通用属性 
      -> 链接器
      -> 输入
      -> 附加依赖项
      库目录 下添加  上述库目录下的  lib文件
      带d的  
       opencv_world320d.lib/opencv_world340d.lib


      也可以将这下lib 复制到 window 操作系统目录下
      C:\Windows\SysWOW64   64位
      C:\Windows\System32   32位
## linux下安装
安装依赖：

        sudo apt-get install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg62-dev libtiff4-dev cmake libswscale-dev libjasper-dev
        
[github源码安装](https://github.com/opencv/opencv.git)

      mkdir build
      cd build
      cmake
      make -j3
      sudo make install
