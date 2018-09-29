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

# RGB相机标定结果：
[参考](https://blog.csdn.net/qingsong1001/article/details/81779236)

    %YAML:1.0
    calibration_time: "2018年09月29日 星期六 10时59分35秒"
    nframes: 40            # 进行 标定的图像数量
    image_width: 640       #  图像的像素 尺寸
    image_height: 480
    board_width: 8         # 水平方向corners数量 
    board_height: 10       # 垂直方向corners数量
    square_size: 1.        # 默认方格的大小
    aspectRatio: 1.
    flags: 2               #calibrateCamera函数进行标定时调整的方式
    camera_matrix: !!opencv-matrix  # 内参数
       rows: 3
       cols: 3
       dt: d
       data: [ 5.5991319678858906e+02, 0., 2.8763263841844133e+02, 0.,
           5.5991319678858906e+02, 2.6363575121251381e+02, 0., 0., 1. ]   # fx 0 cx 0 fy cy 0 0 1

    distortion_coefficients: !!opencv-matrix    # 畸变参数
       rows: 5
       cols: 1
       dt: d
       data: [ -4.4932428999903368e-01, 3.6014186521811137e-01,
           -8.5242709410947044e-04, 1.1130236682130820e-03,
           -2.3177353174684856e-01 ]                                      # k_1,k_2,p_1,p_2,k_3,[k_4,...]
           
## D深度相机标定结果 两个红外相机 
    
    
    
       
