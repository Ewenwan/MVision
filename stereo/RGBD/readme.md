# RGBD 相机
## 图漾 FM810-HD

    在 Linux 环境下执行以下命令下载 SDK。
    gitclone https://github.com/percipioxyz/Camport2.git

    1.  根据 Linux 系统的权限管理机制,默认情况下需要 root 权限才能操作深度摄像头设备,
        非 root用户可以创建 udev rule 来修改设备权限。
        /etc/udev/rules.d/88-tyusb.rules
        规则文件名的开头须为数字(0 ~ 99),数字越大,优先级越高。
    2. camport2/lib/linux 目录下相应平台的文件夹,将文件夹中的 libcamm.so 文件复制到 /usr/lib/ 目
    录下

    3. 进入 camport2 目录,执行以下命令编译实例代码。
    cd sample
    mkdir build
    cd build
    cmake ..
    make


    4. 在 camport2/sample/build/bin 目录下生成若干编译生成的可执行文件,
       以 root 权限运行如下命令,即可查看深度图。
    sudo ./SimpleView_FetchFrame
    
    
    5. 新增 RGB彩色图像采集程序,用于相机标定
       sample/Capature_rgb_Img/main.cpp
       
       修改 sample/CMakeLists.txt：
       set(ALL_SAMPLES
            Capature_rgb_Img  # 新增加
            LoopDetect
            ...
            )
