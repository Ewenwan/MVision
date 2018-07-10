# Parallel Tracking And Mapping (PTAM) 特征点法 fast角点+灰度块匹配 2d-2d单应变换

[本文github](https://github.com/Ewenwan/MVision/tree/master/vSLAM/PTAM)

[项目主页](http://www.robots.ox.ac.uk/~gk/PTAM/)

[github 代码 makefile工程改成了cmake工程](https://github.com/Ewenwan/PTAM4AR)


![](http://image.mamicode.com/info/201802/20180211193608683439.png)
      
      PTAM[1]是视觉SLAM领域里程碑式的项目。
      在此之前，MonoSLAM[2]为代表的基于卡尔曼滤波的算法架构是主流，
      它用单个线程逐帧更新相机位置姿态和地图。
      
      
      2007年，Klein等人提出了PTAM（Parallel Tracking and Mapping），
      这也是视觉SLAM发展过程中的重要事件。
      
      PTAM的重要意义在于以下两点：
      
      1、PTAM提出并实现了 跟踪 与 建图 过程的并行化。
         我们现在已然清楚，跟踪部分需要实时响应图像数据，而对地图的优化则没必要实时地计算。
         后端优化可以在后台慢慢进行，然后在必要的时候进行线程同步即可。
         这是视觉SLAM中首次区分出前后端的概念，引领了后来许多视觉SLAM系统的设计（我们现在看到的SLAM多半都分前后端）。
         
      2、PTAM是第一个使用 非线性优化，而不是使用传统的 滤波器 作为后端的方案。
         它引入了关键帧机制：
               我们不必精细地处理每一幅图像，而是把几个关键图像串起来，然后优化其轨迹和地图。
               早期的SLAM大多数使用EKF滤波器或其变种，以及粒子滤波器等；
               在PTAM之后，视觉SLAM研究逐渐转向了以非线性优化为主导的后端。
               由于之前人们未认识到后端优化的稀疏性，
               所以觉得优化后端无法实时处理那样大规模的数据，而PTAM则是一个显著的反例。
      
      根据PTAM估计的相机位姿，我们可以在一个虚拟的平面上放置虚拟物体，看起来就像在真实的场景中一样。
      
      
      PTAM是增强现实相关的一个库，与SLAM框架类似，兼容ROS。
      ORB-SLAM就是继承并改进PTAM的
      
      PTAM，全称Parallel Tracking And Mapping，
      是最早提出将Track和Map分开作为两个线程的一种SLAM算法，
      是一种基于关键帧的单目视觉SLAM算法。
      
      PTAM（Parallel Tracking and Mapping）架构更多的是系统上的设计，
      姿态跟踪（Tracking）和建立地图（Mapping）两个线程是并行的，这实质上是一种针对SLAM的多线程设计。
      PTAM在当前SLAM领域看来是小儿科，但在当时是一个创举，
      第一次让大家觉得对地图的优化可以整合到实时计算中，并且整个系统可以跑起来。
      
      具体而言: 
         1. 姿态跟踪线程 不修改地图，只是利用已知地图来快速跟踪；
         2. 而在建立 地图线程 专注于地图的建立、维护和更新。
         
      即使建立地图线程耗时稍长，姿态跟踪线程仍然有地图可以跟踪（如果设备还在已建成的地图范围内）.
      这是两个事情并行来做的一个好处，但很现实的问题是如果地图建立或优化过慢，
      跟踪线程很容易会因为没有最新的地图或者没有优化过的地图而跟丢。
      
      另外比较实际的工程问题是地图线程的最新地图数据应该lock 
      还是 copy data between threads
      以及threading的实现质量。
      
      PTAM主要分为这几部分：
            1) Track线程
                1. 金字塔分层，FAST特征提取
                   (对图片构造金字塔的目的有两个：1）加快匹配；2）提高地图点相对于相机远近变化时的鲁棒性)
                   (FAST是常用的特征点，优点是快，缺点是不鲁棒.通常先提取出大量匹配点，后使用SSD快匹配，剔除误匹配)
                2. 地图初始化
                3. 跟踪定位
               　　（极线几何与极线搜索，
               　　  RANSAC（随机采样一致）及N点算法（主要围绕5点算法））
                4. 选取添加关键帧到缓存队列
                5. 重定位(每帧高斯模糊小图SSD相似度匹配)
            2) Map线程
                5.5 先五点法加RANSAC求出初值
                6. 局部BundleAdjustment
                7. 全局BundleAdjustment
                8. 从缓存队列取出关键帧到地图
                9. 极线搜索加点到地图
                
      另一方面，按照一般的视觉SLAM框架，PTAM也可分为
            1) 传感器数据获取（摄像头输入图像数据）
            2) 前端视觉里程计（跟踪定位、重定位）
            3) 后端优化（Bundle Adjustment）
            4) 建图（极线搜索加点）
            5) 没有回环检测

## 主要原理是: 
      从摄影图像上捕捉特征点，
      然后检测出平面，
      在检测出的平面上建立虚拟的3D坐标，
      然后合成摄影图像和CG。
      其中，独特之处在于，立体平面的检测和图像的合成采用并行处理。

## 优点：
      提出并实现了跟踪与建图过程的并行化，
      将前后端分离，使用非线性优化方案，
      既可以实时的定位与建图，
      也可以在虚拟平面上叠加物体。

## 缺点：
      场景小，跟踪容易丢失。
      
## 1. FAST特征提取
    为了SLAM的实时性，选择FAST作为特征提取的方法；
    对于从数据流中输入的每一帧图像先进行金字塔分层(4层均值金字塔)，
    每一层都要进行FAST特征提取；
    像素点半径r的圆周上的点都比当前点像素值大，就是角点。
    
## 2. 地图初始化 单应矩阵Ｈ求解初始变换矩阵 BA 3D-2D优化
      对于每帧图像提取的FAST特征点，因经常出现“扎堆”现象，
      再进行非极大值抑制NMS剔除重复的点，留下较好的特征点，
      然后对每个特征点计算Shi-Tomas得分，
      选出得分较高的特征点（不超过1000个，设置数量阈值），
      作为特征匹配的候选特征点。

      1. 单应变换计算初始位姿
            先选择一帧图像，再通过基于SSD的块匹配选出第二帧图像，作为两帧关键帧；
            根据两帧图像间的匹配特征点，
            计算出两帧间的单应性矩阵Ｈ，
            然后分解出对应的旋转平移矩阵，
            作为相机的初始位姿。 

      2. 设置尺度
            因单目的尺度不确定问题，根据经验设定一个尺度，
            作用于初始两帧间的旋转平移矩阵，并作为全局的尺度。

      3. 三角化计算深度及3d世界点三维坐标
            根据初始两帧间的旋转平移矩阵和特征点像素坐标，
            利用线性三角法深度估计算法估算出第一帧坐标系下的世界点三维坐标，

      4. BA优化3D-2D匹配点误差得到优化后的位姿
            再通过BundleAdjustment方法对世界点和相机初始位姿进行优化；

      5. 极线搜索增添世界点
            因先前计算出的世界点数量可能不够，再通过极线搜索添加世界点，
            再通过BundleAdjustment方法对世界点和相机初始位姿进行优化。

      6. 计算主平面(AR/VR部分)
            根据现在的世界点，通过RANSAC找出主平面(PCL中平面拟合)，
            作为系统的世界坐标系，同时计算出质心C；
            计算出内点和主平面质心的协方差矩阵，通过PCA主成分分析得出主平面的法向量N，

      7. 计算相机和主平面的变换矩阵R,t
            然后通过Gram-Schmidt正交化计算出第一帧坐标系和主平面坐标系旋转矩阵R，
            再根据质心C 和 公式:
               Pw　=　R( Pc − C) = R * Pc − R * C　＝　R * Pc + t
            计算出平移向量 t = − R*C

      8. 统一世界坐标系
            根据主平面计算出的旋转平移矩阵，
            将第一帧坐标系下的世界点和两帧的旋转平移矩阵变换到主平面对应的世界坐标系下，
            第二帧对应的旋转平移矩阵作为当前相机的位姿。
            
## 3. 跟踪定位 3d->2d 最小化重投影误差GN算法
      1）根据上一帧的相机位姿（旋转平移矩阵），
      　 通过运动模型 和 基于 ESM的视觉跟踪算法 对当前帧的相机位姿进行预测。
        
      2）根据预测的相机位姿，将当前所有世界点根据小孔成像原理进行投影，
         投影后的像素点记为pi，并计算出对应的金字塔层级。
         
      3）根据金字塔高层优先原则，选取一定数量世界点（通常，粗搜索选取30～60个，细搜索选取1000个左右）.
      
      4）遍历选取的世界点，对于每一个世界点对应源帧图像中已经进行Warp变换的8x8像素的模板块，
      　 (可能形变较大, 先计算一个 affine warp matrix 变换，将地图点的 patch 变换到和当前帧的相似，
          再在当前帧中用 SSD 算法找匹配的 patch。)
         和以当前帧图像点pi一定范围内的每一个FAST特征点为中心选取的8x8像素块，
         进行基于SSD的相似度计算，选择具有最小SSD值的FAST特征点，
         并记录查找到的特征点数量，用于后期跟踪质量评估；
         出于精确考虑，可通过反向合成图像对齐算法求取该特征点的亚像素坐标，记为pi'，
         这样选取的每个世界点都对应pi和pi'，重投影误差即为p′i−pi. 光度不变性。
         
      5）根据重投影误差建立起误差函数，以预测的相机位姿作为初始值，
      　 通过高斯-牛顿非线性优化方法计算出当前帧的相机位姿，
         其中每次迭代的位姿增量为李代数形式。
         
     实际上，PTAM中位姿计算分粗跟踪和细跟踪两个阶段，
     每个阶段均进行上述的（3）～（5）过程，主要差别在于选取世界点进行计算的点数。
     
## 4. 关键帧选取
    关键帧选取的指标主要有：
      1. 跟踪质量（主要根据跟踪过程中搜索到的点数和搜索的点数比例）
      2. 距离最近关键帧的距离是否足够远（空间）
      3. 距离上一关键帧的帧数是否足够多（时间）
      4. 关键帧缓存队列是否已满

## 5. 重定位
      构建关键帧时，每一帧都会生成一个高斯模糊小图
      （大小为顶层金字塔尺寸的一半，并进行了0.75的高斯卷积，以及去中心化）。
      1. 搜索最相似的参考帧
         重定位时，基于SSD算法 计算 当前帧的高斯模糊小图 和 
         地图中所有的关键帧的高斯模糊小图 的 相似度，
      　 选择相似度最高的关键帧的相机位姿，
      2. 计算变换位姿
      　 根据基于ESM的视觉跟踪算法计算出相机位姿作为当前帧的相机位姿。
        
## 6. 光束法平差BA 最小化重投影误差
      Bundle Adjustment（以下简称BA），中文翻译“光束法平差”，本质是一个优化模型，
      目的是最小化重投影误差，用于最后一步优化，优化相机位姿和世界点。
      PTAM中BA主要在Map线程中，分为局部BA和全局BA，是其中比较耗时的操作。

      1. 局部BA用于优化局部的相机位姿，提高跟踪的精确度；
      2. 全局BA用于全局过程中的相机位姿，使相机经过长时间、长距离的移动之后，相机位姿还比较准确。
      BA是一个图优化模型，一般选择LM(Levenberg-Marquardt)算法并在此基础上利用BA模型的稀疏性进行计算；
      可以直接计算，也可以使用g2o或者Ceres等优化库进行计算。
      
## 7. 极线搜索
      选择关键帧容器中最后一帧作为源帧，然后在所有关键帧中找到距离其最近的一帧作为目标帧。
      通过源帧中像素特征点、场景平均深度和场景深度方差，根据对极几何原理，
      找出源帧中平均深度附近一定范围光束，
      并将其投影到目标帧成像平面，为一段极线。

      遍历该段极线附近所有的候选特征点，
      通过基于SSD的块匹配方法查找出与源帧图像匹配的特征点，
      再通过反向合成算法求取其亚像素坐标，然后三角法计算世界点。
      
## 8. 总结  
      优化算法：
            基本是基于最小二乘的非线性优化算法，跟踪部分使用G-N求解基于权重的最小二乘，BA使用L-M；
      矩阵求逆：
            Cholesky分解；
      线性方程组求解：
            SVD分解 ；
      初始化相机位姿求解：
            用的是基于2D-2D的对极几何计算单应性矩阵，适用于共面特征点；
            后面跟踪部分相机位姿求解，用的是基于2D-3D的PnP算法，P3P/P6P；
      块匹配：基于SSD的相似度计算，而没有用特征进行匹配，
             和SVO 一样，用的是 patch（8x8 方块）,计算相似度匹配时，需要先进行仿射变换，形状变化了。
          例如ORB-SLAM使用ORB特征进行特征匹配计算位姿，回环检测和重定位。
 
[PTAM之Mapping线程 对极几何　单应矩阵　主平面　PCA分解求法线　](https://blog.csdn.net/ilotuo/article/details/51831010)
 
[PTAM之姿态估计 ](https://blog.csdn.net/ilotuo/article/details/51830928)
   
   
## 9. 代码分析
[依赖库安装](https://github.com/Ewenwan/MVision/blob/master/vSLAM/PTAM/3rdParty_install.sh)

      下载代码并安装： 
      git clone https://github.com/Ewenwan/PTAM4AR.git
      cd PTAM4AR
      mkdir build
      cd build
      cmake ..
      make -j
### 主要依赖库
#### libCVD (computer vision library) - 计算机视觉库，主要用于计算机视觉和视频、图片处理
      fast_corner fast角点检测
      video yuv420 yuv411编解码  视频读取 
[代码](https://github.com/Ewenwan/libcvd)

#### GVars3 (configuration system library) - 系统配置库，属于libCVD的子项目，功能是读取配置文件，获取命令行数
      GUI  简单 用户界面程序
      GV3::get() 获取配置参数
[代码](https://github.com/Ewenwan/gvars)


#### TooN (Tom’s Object-oriented numerics library) - 主要用于大量小矩阵的运算，尤其是矩阵分解和优化

### 代码结构分析
[代码分析参考](https://blog.csdn.net/aquathinker/article/details/7768519)

      入口函数 src/main.cc
            //GVars3::GUI
            GUI.LoadFile("../config/settings.cfg");// 载入相机参数等配置文件
            System s;// 创建系统对象 自动执行 System::System()函数
            s.Run(); // 运行
            
src/System.cc
#### A. 系统对象构造函数 System::System()
      0. 对象继承于
         mpVideoSource(new VideoSourceV4L())      // 视频处理对象 V4L库视频对象 src/VideoSource.cc
         mGLWindow(mpVideoSource->Size(), "PTAM") // 菜单 GLWindow2 mGLWindow      src/GLWindow2.cc  

      1. 注册一系列命令、添加相对应的功能按钮。
          //GVars3::GUI
          GUI.RegisterCommand("exit", GUICommandCallBack, this);// 退出
          GUI.RegisterCommand("quit", GUICommandCallBack, this);// 停止

      2. 检查相机参数是否已经传入，否则退出，去进行相机标定
          使用GVars3库函数
          vTest = GV3::get<Vector<NUMTRACKERCAMPARAMETERS> >("Camera.Parameters", ATANCamera::mvDefaultParams, HIDDEN);

      3. 创建摄像机ATANCamera对象 
          // ATANCamera.cc 相机内参数、畸变参数、图像大小、归一化平面投影
          mpCamera = new ATANCamera("Camera");

      4. 创建地图Map 地图创建管理器MapMaker 跟踪器Tracker 增强现实AR驱动ARDriver 地图显示MapViewer 
            mpMap = new Map;                              // src/Map.cc      地图
            mpMapMaker = new MapMaker(*mpMap, *mpCamera); // src/MapMaker.cc 地图管理器
            mpTracker = new Tracker(mpVideoSource->Size(), *mpCamera, *mpMap, *mpMapMaker);// src/Tracker.cc   跟踪器
            mpARDriver = new ARDriver(*mpCamera, mpVideoSource->Size(), mGLWindow);        // src/ARDriver.cc  虚拟物体
            mpMapViewer = new MapViewer(*mpMap, mGLWindow);                                // src/MapViewer.cc 地图显示
      5. 初始化GUI游戏菜单及相应功能按钮。
            GUI.ParseLine("GLWindow.AddMenu Menu Menu");
            GUI.ParseLine("Menu.ShowMenu Root");
            GUI.ParseLine("Menu.AddMenuButton Root Reset Reset Root");
            GUI.ParseLine("Menu.AddMenuButton Root Spacebar PokeTracker Root");
            GUI.ParseLine("DrawAR=0");
            GUI.ParseLine("DrawMap=0");
            GUI.ParseLine("Menu.AddMenuToggle Root \"View Map\" DrawMap Root");
            GUI.ParseLine("Menu.AddMenuToggle Root \"Draw AR\" DrawAR Root");
      6. 初始化标志 
            mbDone = false;// 初始化时mbDone = false;  

####   B. 系统运行函数 void System::Run()
      1. 创建图像处理对象
            CVD::Image<CVD::Rgb<CVD::byte> > imFrameRGB(mpVideoSource->Size());// 彩色图像用于最终的显示
            CVD::Image<CVD::byte> imFrameBW(mpVideoSource->Size());//黑白(灰度)图像用于处理追踪相关等功能
      2. 采集上述两种图像
            mpVideoSource->GetAndFillFrameBWandRGB(imFrameBW, imFrameRGB);
      3. 系统跟踪和建图， 更新系统帧
            UpdateFrame(imFrameBW, imFrameRGB);

####  C. 系统跟踪和建图， 更新系统帧 System::UpdateFrame()
      1. 系统初始化，第一帧的处理，单应变换求解3D点云，生成初始地图
      2. 设置可是化窗口相关属性
      3. 读取 显示配置参数
      4. 显示表示更新，DrawMap及DrawAR状态变量的判断
      5. 开始追踪黑白图像(相机位姿跟踪)
             多层级金字塔图像(多金字塔尺度) FAST角点检测匹配跟踪
             每一个层级的阈值有所不同。最后生成按列角点查询表，便于以后近邻角点的查询任务.
            mpTracker->TrackFrame(imBW, !bDrawAR && !bDrawMap);// Tracker::TrackFrame() src/Tracker.cc
      6. 可视化显示点云和 虚拟物体
      7.可视化文字菜单显示
      
跟踪线程主要函数文件 src/Tracker.cc

#### Tracker::TrackFrame() 跟踪每一帧图像

      步骤1： 预处理，为当前关键帧生成4级金字塔图像(多尺度金字塔)，进行FAST角点检测，生成角点查找表
              mCurrentKF.MakeKeyFrame_Lite(imFrame);// src/KeyFrame.cc, KeyFrame::MakeKeyFrame_Lite()
              1. 生成金字塔图像，上一层下采样得到下一层图像
                  lev.im.resize(aLevels[i - 1].im.size() / 2);// 尺寸减半
                  halfSample(aLevels[i - 1].im, lev.im);// 上一层下采样，得到下一层图像

              2. FAST角点 检测，对每一层进行 FAST角点 检测
                  if (i == 0)// 第0层 图像
                        fast_corner_detect_10(lev.im, lev.vCorners, 10);
                  if (i == 1)// 第1层 图像
                        fast_corner_detect_10(lev.im, lev.vCorners, 15);
                  if (i == 2)// 第2层 图像
                        fast_corner_detect_10(lev.im, lev.vCorners, 15);
                  if (i == 3)// 第4层 图像
                        fast_corner_detect_10(lev.im, lev.vCorners, 10);

               3. 建立角点查找表，加快查找，对每一行的角点创建查找表LUT 加速查找邻居角点
      步骤2：更新小图，为估计旋转矩阵做准备
      步骤3：显示图像(第0层)和FAST角点
                  1. 运动模型跟踪上一帧(求解初始位置，上一帧速度乘上上一帧位姿)
                        Tracker::PredictPoseWithMotionModel();

                  2. 跟踪地图 最重要的部分
                        Tracker::TrackMap();
                            a. 地图点根据帧初始位姿和相机参数投影到帧的二维图像平面上，
                               跳过不在相机平面上的点
                            b. patch 匹配查找匹配点对
                            c. 3d-2d p6p求解 

                  3. 更新运动模型(前后两帧变换矩阵)
                        Tracker::UpdateMotionModel();

                  4. 确保跟踪质量
                        Tracker::AssessTrackingQuality();
                  5. 显示更新系统跟踪状态信息(跟踪质量好坏 每一层fast角点熟练 地图点和关键帧数量)
                        manMeasFound[i]；
                        manMeasAttempted[i];
                        mMap.vpPoints.size()；
                        mMap.vpKeyFrames.size()；
                  6. 关键帧判断+创建关键帧   
                        mMapMaker.IsNeedNewKeyFrame(mCurrentKF);//是否需要创建关键帧 MapMaker::IsNeedNewKeyFrame()
                        mMapMaker.AddKeyFrame(mCurrentKF);// 创建关键帧
                  7. 跟踪丢失的处理--类似重定位处理
                        Tracker::AttemptRecovery();// 重定位
                        Tracker::TrackMap();       // 跟踪地图，更新位姿
                  8. 起初地图质量不好(点比较少)，初始地图跟踪
                        Tracker::TrackForInitialMap();
                        
                        
建图线程主要函数文件 src/MapMaker.cc
#### MapMaker::run()
      步骤1. 局部地图优化
             MapMaker::BundleAdjustRecent();
      步骤2. 地图点投影到关键帧，无匹配到的角点，三角化，创建新的地图点
            MapMaker::ReFindNewlyMade();
                MapMaker::Triangulate();
      步骤3. 全局地图优化
            MapMaker::BundleAdjustAll();
      步骤4. 查找外点
            MapMaker::ReFindFromFailureQueue();
      步骤5. 处理外点
            MapMaker::HandleBadPoints();
      步骤6. 添加关键帧到地图
            MapMaker::AddKeyFrameFromTopOfQueue();
