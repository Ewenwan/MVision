# ORB-SLAM2 ORB特征点法SLAM 支持单目、双目、rgbd相机

[安装测试](https://github.com/Ewenwan/MVision/blob/master/vSLAM/oRB_SLAM2/install.md)

[本文github链接](https://github.com/Ewenwan/MVision/blob/master/vSLAM/oRB_SLAM2/readme.md)

    ORB-SLAM是一个基于特征点的实时单目SLAM系统，在大规模的、小规模的、室内室外的环境都可以运行。
    该系统对剧烈运动也很鲁棒，支持宽基线的闭环检测和重定位，包括全自动初始化。
    该系统包含了所有SLAM系统共有的模块：
        跟踪（Tracking）、建图（Mapping）、重定位（Relocalization）、闭环检测（Loop closing）。
    由于ORB-SLAM系统是基于特征点的SLAM系统，故其能够实时计算出相机的轨线，并生成场景的稀疏三维重建结果。
    ORB-SLAM2在ORB-SLAM的基础上，还支持标定后的双目相机和RGB-D相机。
 
**系统框架**

![](https://img-blog.csdn.net/20161114115058814)  

**贡献**

![](https://img-blog.csdn.net/20161114115026626)


# 1. 相关论文：

[ORB-SLAM 单目Monocular特征点法](http://webdiis.unizar.es/%7Eraulmur/MurMontielTardosTRO15.pdf)

[ORB-SLAM2 单目双目rgbd](https://128.84.21.199/pdf/1610.06475.pdf)

[词袋模型DBoW2 Place Recognizer](http://doriangalvez.com/papers/GalvezTRO12.pdf)


> 原作者目录:

[Raul Mur-Artal](http://webdiis.unizar.es/~raulmur/)

[Juan D. Tardos](http://webdiis.unizar.es/~jdtardos/),

[J. M. M. Montiel](http://webdiis.unizar.es/~josemari/) 

[Dorian Galvez-Lopez](http://doriangalvez.com/)

([DBoW2](https://github.com/dorian3d/DBoW2))

# 2. 简介
    ORB-SLAM2 是一个实时的 SLAM  库，
    可用于 **单目Monocular**, **双目Stereo** and **RGB-D** 相机，
    用来计算 相机移动轨迹 camera trajectory 以及稀疏三维重建sparse 3D reconstruction 。
    
    在 **双目Stereo** 和 **RGB-D** 相机 上的实现可以得到真是的 场景尺寸稀疏三维 点云图
    可以实现 实时回环检测detect loops 、相机重定位relocalize the camera 。 
    提供了在  
[KITTI 数据集](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) 上运行的 SLAM 系统实例，支持双目stereo 、单目monocular。

[TUM 数据集](http://vision.in.tum.de/data/datasets/rgbd-dataset) 上运行的实例，支持 RGB-D相机 、单目相机 monocular, 

[EuRoC 数据集](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)支持 双目相机 stereo 、单目相机 monocular.

    也提供了一个 ROS 节点 实时运行处理  单目相机 monocular, 双目相机stereo 以及 RGB-D 相机 数据流。
    提供一个GUI界面 可以来 切换 SLAM模式 和重定位模式

>**支持的模式: 

    1. SLAM Mode  建图定位模式
        默认的模式,三个并行的线程: 跟踪Tracking, 局部建图 Local Mapping 以及 闭环检测Loop Closing. 
        定位跟踪相机localizes the camera, 建立新的地图builds new map , 检测到过得地方 close loops.

    2. Localization Mode 重定位模式
        适用与在工作地方已经有一个好的地图的情况下。执行 局部建图 Local Mapping 以及 闭环检测Loop Closing 两个线程.  
        使用重定位模式,定位相机

# 3. 系统工作原理
    可以看到ORB-SLAM主要分为三个线程进行，
    也就是论文中的下图所示的，
![](https://img-blog.csdn.net/20161114115114018)    
    
    分别是Tracking、LocalMapping和LoopClosing。
    ORB-SLAM2的工程非常清晰漂亮，
    三个线程分别存放在对应的三个文件中，
    分别是：
    Tracking.cpp、
    LocalMapping.cpp　和
    LoopClosing.cpp　文件中，很容易找到。 

## A. 跟踪（Tracking）
      前端位姿跟踪线程采用恒速模型，并通过优化重投影误差优化位姿，
      这一部分主要工作是从图像中提取ORB特征，
      根据上一帧进行姿态估计，
      或者进行通过全局重定位初始化位姿，
      然后跟踪已经重建的局部地图，
      优化位姿，再根据一些规则确定新的关键帧。

## B. 建图（LocalMapping）
      局部地图线程通过MapPoints维护关键帧之间的共视关系，
      通过局部BA优化共视关键帧位姿和MapPoints，这一部分主要完成局部地图构建。
      包括对关键帧的插入，验证最近生成的地图点并进行筛选，然后生成新的地图点，
      使用局部捆集调整（Local BA），
      最后再对插入的关键帧进行筛选，去除多余的关键帧。

## C. 闭环检测（LoopClosing）
      闭环检测线程通过bag-of-words加速闭环匹配帧的筛选，
      并通过Sim3优化尺度，通过全局BA优化Essential Graph和MapPoints，
      这一部分主要分为两个过程：
      分别是：　闭环探测　和　闭环校正。
       闭环检测：　
       　　　　　先使用WOB进行探测，然后通过Sim3算法计算相似变换。
       闭环校正：
       　　　　　主要是闭环融合和Essential Graph的图优化。
      
## D. 重定位 Localization

    使用bag-of-words加速匹配帧的筛选，并使用EPnP算法完成重定位中的位姿估计。
         
         
# 4. 代码分析
[ORB-SLAM2详解（二）代码逻辑](https://blog.csdn.net/u010128736/article/details/53169832)

# 4.1 应用程序框架
>**单目相机app框架：**
```asm
        1. 创建 单目ORB_SLAM2::System SLAM 对象
        2. 载入图片 或者 相机捕获图片 im = cv::imread();
        3. 记录时间戳 tframe ，并计时，
	   std::chrono::steady_clock::now();    // c++11
	   std::chrono::monotonic_clock::now();
        4. 把图像和时间戳 传给 SLAM系统, SLAM.TrackMonocular(im,tframe); 
        5. 计时结束，计算时间差，处理时间。
        6. 循环2-5步。
        7. 结束，关闭slam系统，关闭所有线程 SLAM.Shutdown();
        8. 保存相机轨迹, SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
```
![](https://img-blog.csdn.net/20161115115034740)

>**双目相机app程序框架：**
```asm
	1. 读取相机配置文件(内参数 畸变矫正参数 双目对齐变换矩阵) =======================
	   cv::FileStorage fsSettings(setting_filename, cv::FileStorage::READ);
	   fsSettings["LEFT.K"] >> K_l;//内参数
	   fsSettings["LEFT.D"] >> D_l;// 畸变矫正
	   fsSettings["LEFT.P"] >> P_l;// P_l,P_r --左右相机在校准后坐标系中的投影矩阵 3×4
	   fsSettings["LEFT.R"] >> R_l;// R_l,R_r --左右相机校准变换（旋转）矩阵  3×3
	2. 计算双目矫正映射矩阵========================================================
	   cv::Mat M1l,M2l,M1r,M2r;
           cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
           cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
	3. 创建双目系统=============================================================== 
	   ORB_SLAM2::System SLAM(vocabulary_filepath, setting_filename, ORB_SLAM2::System::STEREO, true);
	4. 从双目设备捕获图像,设置分辨率捕获图像========================================
	   cv::VideoCapture CapAll(deviceid); //打开相机设备 
	   //设置分辨率   1280*480  分成两张 640*480  × 2 左右相机
           CapAll.set(CV_CAP_PROP_FRAME_WIDTH,1280);  
           CapAll.set(CV_CAP_PROP_FRAME_HEIGHT, 480); 
        5. 获取左右相机图像===========================================================
	   CapAll.read(src_img);
	   imLeft  = src_img(cv::Range(0, 480), cv::Range(0, 640));   
           imRight = src_img(cv::Range(0, 480), cv::Range(640, 1280));   
	6. 使用2步获取的双目矫正映射矩阵 矫正 左右相机图像==============================
	   cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
           cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);
	7. 记录时间戳 time ，并计时===================================================
		#ifdef COMPILEDWITHC11
		   std::chrono::steady_clock::time_point    t1 = std::chrono::steady_clock::now();
		#else
		   std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
		#endif
        8. 把左右图像和时间戳 传给 SLAM系统===========================================
	   SLAM.TrackStereo(imLeftRect, imRightRect, time);
	9. 计时结束，计算时间差，处理时间============================================= 
	10.循环执行 5-9步              =============================================
	11.结束，关闭slam系统，关闭所有线程===========================================   
	12.保存相机轨迹                   ========================================== 
```
>**ORB_SLAM2::System SLAM 对象框架:**
```asm
        在主函数中，我们创建了一个ORB_SLAM2::System的对象SLAM，这个时候就会进入到SLAM系统的主接口System.cc。
        这个代码是所有调用SLAM系统的主入口，
        在这里，我们将看到前面博客所说的ORB_SLAM的三大模块：
        Tracking、LocalMapping 和 LoopClosing。

    System类的初始化函数：
        1. 创建字典 mpVocabulary = new ORBVocabulary()；并从文件中载入字典 
           mpVocabulary = new ORBVocabulary();           // 创建关键帧字典数据库
           // 读取 txt格式或者bin格式的 orb特征字典, 
           mpVocabulary->loadFromTextFile(strVocFile);   // txt格式
           mpVocabulary->loadFromBinaryFile(strVocFile); // bin格式
           
        2. 使用特征字典mpVocabulary 创建关键帧数据库 
           mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);
           
        3. 创建地图对象 mpMap 
           mpMap = new Map();
           
        4. 创建地图显示(mpMapDrawer) 帧显示(mpFrameDrawer) 两个显示窗口   
           mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);//地图显示
           mpFrameDrawer = new FrameDrawer(mpMap);//关键帧显示
           
        5. 初始化 跟踪线程(mpTracker) 对象 未启动
	       mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
				      mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor); 
                      
        6. 初始化 局部地图构建线程(mptLocalMapping) 并启动线程
	       mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
	       mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);
          
        7. 初始化 闭环检测线程(mptLoopClosing) 并启动线程
	       mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
	       mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);
          
        8. 初始化 跟踪线程可视化 并启动
           mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
		   mptViewer = new thread(&Viewer::Run, mpViewer);
		   mpTracker->SetViewer(mpViewer);
           
        9. 线程之间传递指针 Set pointers between threads
	       mpTracker->SetLocalMapper(mpLocalMapper);   // 跟踪线程 关联 局部建图和闭环检测线程
	       mpTracker->SetLoopClosing(mpLoopCloser);
	       mpLocalMapper->SetTracker(mpTracker);       // 局部建图线程 关联 跟踪和闭环检测线程
	       mpLocalMapper->SetLoopCloser(mpLoopCloser);
	       mpLoopCloser->SetTracker(mpTracker);        // 闭环检测线程 关联 跟踪和局部建图线程
	       mpLoopCloser->SetLocalMapper(mpLocalMapper);
```     
        如下图所示： 
![](https://img-blog.csdn.net/20161115115045032)

>**单目跟踪SLAM.TrackMonocular()框架**
```asm
	1. 模式变换的检测  跟踪+建图  or  跟踪+定位+建图
	2. 检查跟踪tracking线程重启
	3. 单目跟踪
	   mpTracker->GrabImageMonocular(im,timestamp);// Tracking.cc中
	   // 图像转换成灰度图，创建帧Frame对象(orb特征提取等)
	   // 使用Track()进行跟踪: 
	                 0. 单目初始化(最开始执行) MonocularInitialization();// 单目初始化
	                 a. 两帧跟踪得到初始化位姿(跟踪上一帧/跟踪参考帧/重定位)
	                 b. 跟踪局部地图，多帧局部地图G2O优化位姿, 新建关键帧
```
>**双目跟踪System::TrackStereo()框架**
```asm
	1. 模式变换的检测  跟踪+建图  or  跟踪+定位+建图
	2. 检查跟踪tracking线程重启
	3. 双目跟踪
	   mpTracker->GrabImageStereo(imLeft,imRight,timestamp); // Tracking.cc中
	   // 图像转换成灰度图，创建帧Frame对象(orb特征提取器,分块特征匹配，视差计算深度)
	   // 使用Track()进行跟踪:
	                 0. 双目初始化(最开始执行) StereoInitialization();// 双目 / 深度初始化
	                 a. 两帧跟踪得到初始化位姿(跟踪上一帧/跟踪参考帧/重定位)
		         b. 跟踪局部地图，多帧局部地图G2O优化位姿, 新建关键帧
```
>**深度相机跟踪System::TrackRGBD()框架**
```asm
	1. 模式变换的检测  跟踪+建图  or  跟踪+定位+建图
	2. 检查跟踪tracking线程重启
	3. 双目跟踪
	   mpTracker->GrabImageRGBD(im,depthmap,timestamp); // Tracking.cc中
	   // 图像转换成灰度图，创建帧Frame对象(orb特征提取器,深度图初始化特征点深度)
	   // 使用Track()进行跟踪: 
	                 0. RGBD初始化(最开始执行) StereoInitialization();// 双目 / 深度初始化
	                 a. 两帧跟踪得到初始化位姿(跟踪上一帧/跟踪参考帧/重定位)
		         b. 跟踪局部地图，多帧局部地图G2O优化位姿, 新建关键帧
```


# 4.2 单目/双目/RGBD初始化, 单应变换/本质矩阵恢复3d点
[参考](https://blog.csdn.net/u010128736/article/details/53218140)

## 1. 单目初始化      Tracking::MonocularInitialization()
    系统的第一步是初始化，ORB_SLAM使用的是一种自动初始化方法。
    这里同时计算两个模型：
	1. 用于 平面场景的单应性矩阵H( 4对 3d-2d点对，线性方程组，奇异值分解) 
	2. 用于 非平面场景的基础矩阵F(8对 3d-2d点对，线性方程组，奇异值分解)，

[推到 参考 单目slam基础](https://github.com/Ewenwan/MVision/blob/master/vSLAM/%E5%8D%95%E7%9B%AEslam%E5%9F%BA%E7%A1%80.md)

    然后通过一个评分规则来选择合适的模型，恢复相机的旋转矩阵R和平移向量t。
    函数调用关系：
	Tracking::GrabImageMonocular() 创建帧对象(第一帧提取orb特征点数量较多,为后面帧的两倍) -> 
	Tracking::Track() ->  初始化 MonocularInitialization();// 单目初始化
		  |  1. 第一帧关键点个数超过 100个，进行初始化  mpInitializer = new Initializer(mCurrentFrame,1.0,200)；
		  |  2. 第二帧关键点个数 小于100个，删除初始化器,跳到第一步重新初始化。
		  |  3. 第二帧关键点个数 也大于100个(只有连续的两帧特征点 均>100 个才能够成功构建初始化器)
		  |     构建 两两帧 特征匹配器    ORBmatcher::ORBmatcher matcher(0.9,true)
		  |     金字塔分层块匹配搜索匹配点对  100为搜索窗口大小尺寸尺度 
		  |     int nmatches=matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);
		  |  4. 如果两帧匹配点对过少(nmatches<100)，跳到第一步，重新初始化。
		  |  5. 匹配点数量足够多(nmatches >= 100),进行单目初始化：
		  |     第一帧位置设置为：单位阵 Tcw = cv::Mat::eye(4,4,CV_32F);
		  |     使用 单应性矩阵H 和 基础矩阵F 同时计算两个模型，通过一个评分规则来选择合适的模型，
		  |     恢复第二帧相机的旋转矩阵Rcw 和 平移向量 tcw，同时 三角变换得到 部分三维点 mvIniP3D
		  |     mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
		  |  6. 设置初始参考帧的世界坐标位姿态:
		  |     mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
		  |  7. 设置第二帧(当前帧)的位姿
		  |     cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
		  |     Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
		  |     tcw.copyTo(Tcw.rowRange(0,3).col(3)); 
		  |     mCurrentFrame.SetPose(Tcw);
		  |  8. 创建初始地图 使用 最小化重投影误差BA 进行 地图优化 优化位姿 和地图点
		  |     CreateInitialMapMonocular();
		  |
		  |-> 后面帧的跟踪 -> 两两帧的跟踪得到初始位姿
			  |  1. 有运动速度，使用恒速运动模式跟踪上一帧   Tracking::TrackWithMotionModel()
			  |  2. 运动量小或者 1.跟踪失败，跟踪参考帧模式  Tracking::TrackReferenceKeyFrame()
			  |  3. 1 和 2都跟踪失败的话，使用重定位跟踪，跟踪可视参考帧群 Tracking::Relocalization()
			  | 
		         之后 -> 跟踪局部地图(一级二级相连关键帧), 使用图优化对位姿进行精细化调整 Tracking::TrackLocalMap()
			     |
			     -> 判断是否需要新建关键帧   Tracking::NeedNewKeyFrame();  Tracking::CreateNewKeyFrame();
			         |
				 |-> 跟踪失败后的处理

> 单目位姿恢复分析 Initializer::Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated)) 

    步骤1：根据 matcher.SearchForInitialization 得到的初始匹配点对，筛选后得到好的特征匹配点对
    步骤2：在所有匹配特征点对中随机选择8对特征匹配点对为一组，共选择mMaxIterations组
    步骤3：调用多线程分别用于计算fundamental matrix(基础矩阵F) 和 homography(单应性矩阵)
           Initializer::FindHomography();   得分为SH
	   Initializer::FindFundamental();  得分为SF
    步骤4：计算评价得分 RH，用来选取某个模型
           float RH = SH / (SH + SF);// 计算 选着标志
    步骤5：根据评价得分，从单应矩阵H 或 基础矩阵F中恢复R,t
	   if(RH>0.40)// 更偏向于 平面  使用  单应矩阵恢复
	       return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
	   else //if(pF_HF>0.6) // 偏向于非平面  使用 基础矩阵 恢复
	       return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

### a. fundamental matrix(基础矩阵F) Initializer::FindHomography()

### b. homography(单应性矩阵)  Initializer::FindFundamental()


### c. 基础矩阵F恢复R,t  Initializer::ReconstructF()

### d. 单应矩阵H恢复R,t  Initializer::ReconstructH()
	
### e. 创建初始地图      CreateInitialMapMonocular();

## 2. 双目/RGBD初始化 Tracking::StereoInitialization()


# 4.3 跟踪线程 Tracking 两帧跟踪/局部地图跟踪
[参考](https://blog.csdn.net/u010128736/article/details/53339311)


# 4.4 局部跟踪线程 LocalMapping
[参考](https://blog.csdn.net/u010128736/article/details/53395936)


# 4.5 闭环检测线程 LoopClosing
[参考](https://blog.csdn.net/u010128736/article/details/53409199)

 
# 5. 数学理论总结
[参考]()


