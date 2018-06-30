# ORB-SLAM2 ORB特征点法SLAM 支持单目、双目、rgbd相机

[安装测试](https://github.com/Ewenwan/MVision/blob/master/vSLAM/oRB_SLAM2/install.md)

[本文github链接](https://github.com/Ewenwan/MVision/blob/master/vSLAM/oRB_SLAM2/readme.md)

[orbslam2 + imu](https://github.com/Ewenwan/LearnVIORB)

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
	1. 模式变换的检测  跟踪+定位  or  跟踪+定位+建图
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
	1. 模式变换的检测  跟踪+定位  or  跟踪+定位+建图
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
	1. 模式变换的检测  跟踪+定位  or  跟踪+定位+建图
	2. 检查跟踪tracking线程重启
	3. 双目跟踪
	   mpTracker->GrabImageRGBD(im,depthmap,timestamp); // Tracking.cc中
	   // 图像转换成灰度图，创建帧Frame对象(orb特征提取器,深度图初始化特征点深度)
	   // 使用Track()进行跟踪: 
	                 0. RGBD初始化(最开始执行) StereoInitialization();// 双目 / 深度初始化
	                 a. 两帧跟踪得到初始化位姿(跟踪上一帧/跟踪参考帧/重定位)
		         b. 跟踪局部地图，多帧局部地图G2O优化位姿, 新建关键帧
```


# 4.2 单目/双目/RGBD初始化, 单应变换/本质矩阵恢复3d点，创建初始地图
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
		  |     Tracking::CreateInitialMapMonocular()
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
				 
				 
> 初始化时，orb匹配点的搜索 ORBmatcher::SearchForInitialization()

	金字塔分层分块orb特征匹配，然后通过三个限制，滤除不好的匹配点对:
		1. 最小值阈值限制；
		2. 最近匹配距离一定比例小于次近匹配距离；
		3. 特征方向误差一致性判断(方向差直方图统计，保留3个最高的方向差一致性最多的点，一致性较大)
		   ORBmatcher::ComputeThreeMaxima(); // 统计数组中最大 的几个数算法，参考性较大
		   ORBmatcher.cc 1912行 优秀的算法值得参考
	步骤:
		步骤1：为帧1的每一个关键点在帧2中寻找匹配点(同一金字塔层级，对应位置方块内的点，多个匹配点)
		       Frame::GetFeaturesInArea()
		步骤2：计算 1对多 匹配点对描述子之间的距离(二进制变量差异)
		       ORBmatcher::DescriptorDistance(d1,d2); // 只计算了前八个 二进制位 的差异
		       ORBmatcher.cc 1968行 优秀的算法值得参考
		步骤3：保留最小和次小距离对应的匹配点
		       ORBmatcher.cc 570行  优秀的算法值得参考
		步骤4：确保最小距离小于阈值 50
		步骤5：确保最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱，并统计方向差值直方图
		步骤6：特征方向误差一致性判断(方向差直方图统计，保留3个最高的方向差一致性最多的点，一致性较大)
		步骤7：更新匹配信息，用最新的匹配更新之前记录的匹配
	
> 单目位姿恢复分析 Initializer::Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated)) 

    步骤1：根据 matcher.SearchForInitialization 得到的初始匹配点对，筛选后得到好的特征匹配点对
    步骤2：在所有匹配特征点对中随机选择8对特征匹配点对为一组，共选择 mMaxIterations 组
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

### 0. 2D-2D配点对求变换矩阵前先进行标准化处理去 均值后再除以绝对矩 Initializer::Normalize()
```asm
步骤1：计算两坐标的均值
       mean_x  =  sum(ui) / N ；
       mean_y  =  sum(vi) / N；
步骤2：计算绝对局倒数
      绝对矩：
	 mean_x_dev = sum（abs(ui - mean_x)）/ N ；
	 mean_y_dev = sum（abs(vi - mean_y)）/ N ； 
      绝对矩倒数：
	 sX = 1/mean_x_dev 
	 sY = 1/mean_y_dev
步骤3：计算标准化后的点坐标
	ui' = (ui - mean_x) * sX
	vi' = (vi - mean_y) * sY 
步骤4：计算并返回标准化矩阵 
	ui' = (ui - mean_x) * sX =  ui * sX + vi * 0  + (-mean_x * sX) * 1
	vi' = (vi - mean_y) * sY =  ui * 0  + vi * sY + (-mean_y * sY) * 1   
	1   =                       ui * 0  + vi * 0  +      1         * 1

	可以得到：
		ui'     sX  0   (-mean_x * sX)      ui
		vi' =   0   sY  (-mean_y * sY)   *  vi
		1       0   0        1              1
	标准化后的的坐标 = 标准化矩阵T * 原坐标
	所以标准化矩阵:
		T =  sX  0   (-mean_x * sX) 
		     0   sY  (-mean_y * sY)
		     0   0        1 
	而由标准化坐标 还原 回 原坐标(左乘T 逆)：
		原坐标 = 标准化矩阵T 逆矩阵 * 标准化后的的坐标

	再者用于还原 单应矩阵 下面需要用到：
	  p1'  ------> Hn -------> p2'   , p2'   = Hn*p1'
	  T1*p1 -----> Hn -------> T2*p2 , T2*p2 = Hn*(T1*p1)
	  左乘 T2逆 ，得到   p2 = T2逆 * Hn*(T1*p1)= H21i*p1
	  H21i = T2逆 * Hn * T1
```
### a. fundamental matrix(基础矩阵F)  随机采样 找到最好的 基础矩阵 Initializer::FindFundamental() 
```asm
思想：
计算基础矩阵F,随机采样序列8点法,采用归一化的直接线性变换（normalized DLT）求解，
极线几何约束(current frame 1 变换到 reference frame 2)，
在最大迭代次数内调用 ComputeF21计算F,
使用 CheckFundamental 计算此 基础矩阵的得分，
在最大迭代次数内保留最高得分的基础矩阵F。

步骤：
步骤1：将两帧上对应的2d-2d匹配点对进行归一化
	Initializer::Normalize(mvKeys1,vPn1, T1);//  mvKeys1原坐标 vPn1归一化坐标，T1标准化矩阵
	Initializer::Normalize(mvKeys2,vPn2, T2);// 
	cv::Mat T2inv = T2.inv();// 标准化矩阵 逆矩阵
步骤2：在最大迭代次数mMaxIterations内，从标准化后的点中随机选取8对点对
	int idx = mvSets[it][j];//随机数集合 总匹配点数范围内
	vPn1i[j] = vPn1[mvMatches12[idx].first];
	vPn2i[j] = vPn2[mvMatches12[idx].second]; 
步骤3：通过标准化逆矩阵 和 标准化点对的基础矩阵 计算原点对的 基础矩阵   
	cv::Mat Fn = ComputeF21(vPn1i,vPn2i);// 计算 标准化后的点对 对应的 基础矩阵 Fn
	cv::Mat T2t = T2.t(); // 标准化矩阵的 转置矩阵
	 推到：
	 x2 = R*x1 + t
	 t 叉乘 x2 = t 叉乘 R*x1
	 x2转置 * t 叉乘 x2 = x2转置 * t 叉乘 R*x1 = 0
	 得到： 
	 x2转置 * t 叉乘 R  * x1 = x2转置 * R21  * x1 =  0
	 (K逆*p2)转置 * t 叉乘 R  * (K逆*p1) = 
	 p2转置 * K转置逆 * t 叉乘 R  * K逆 * p1 = p2转置 * F21 *  p1

	 上面求到的是 归一化后的点对的变换矩阵：
	 p2'转置 * Fn * p1' = 0
	 (T2*p2)转置 * Fn * (T1 * p1) = 0
	 p2转置 * T2转置 * Fn * T1 * p1 = 0
	 所以得到：
	   未归一化点对对应的变换矩阵F21为：
	   F21 =  T2转置 * Fn * T1 = T2t * Fn * T1

步骤4：通过计算重投影误差来计算单应矩阵的好坏，得分 
	currentScore =Initializer::CheckFundamental(); 
步骤5：保留迭代中，得分最高的单应矩阵和对应的得分
	if(currentScore > score)//此次迭代 计算的单应H的得分较高
	{
	    F21 = F21i.clone();// 保留最优的 基础矩阵 F
	    vbMatchesInliers = vbCurrentInliers;//对应的匹配点对   标记内点
	    score = currentScore;// 最高的得分
	}
```
#### Initializer::ComputeF21(vPn1i,vPn2i) 8对点求解 基础矩阵F
```asm
 推到：
 x2 = R*x1 + t
 t 叉乘 x2 = t 叉乘 R*x1
 x2转置 * t 叉乘 x2 = x2转置 * t 叉乘 R*x1 = 0
 得到： 
 x2转置 * t 叉乘 R  * x1 = x2转置 * R21  * x1 =  0
 (K逆*p2)转置 * t 叉乘 R  * (K逆*p1) = 
 p2转置 * K转置逆 * t 叉乘 R  * K逆 * p1 = p2转置 * F21 *  p1

 一对2d-2d点对 p1-p2得到:
	 p2转置 * F21 *  p1
写成矩阵形式：
             |f1   f2   f3|     |u1|
|u2 v2 1| *  |f4   f5   f6|  *  |v1|    = 0 , 应该=0 不等于零的就是误差
             |f7   f8   f9|     |1|
前两项展开得到：
	a1 = f1*u2 + f4*v2 + f7;
	b1 = f2*u2 + f5*v2 + f8;
	c1 = f3*u2 + f6*v2 + f9;
得到:
		     |u1|
	|a1 b1 c1| * |v1| = 0
		     |1|
得到:
       a1*u1+ b1*v1 + c1= 0
一个点对 得到一个约束方程:
  f1*u1*u2 + f2*v1*u2  + f3*u2 + f4*u1*v2  + f5*v1*v2 + f6*v2 +  f7*u1 + f8*v1 + f9 = 0
写成矩阵形式：
 |u1*u2 v1*u2 u2 u1*v2 v1*v2 v2 u1 v1 1| * |f1 f2 f3 f4 f5 f6 f7 f8 f9|转置 = 0
采样8个点对 可以到八个约束, f 9个参数，8个自由度，另一个为尺度因子(单目尺度不确定的来源)
线性方程组 求解 A * f = 0  
对矩阵A进行奇异值分解得到f ,

对A进行SVD奇异值分解 [U,S,V]=svd(A)，其中U和V代表二个相互正交矩阵，而S代表一对角矩阵
cv::SVDecomp(A,S,U,VT,SVD::FULL_UV);  //后面的FULL_UV表示把U和VT补充称单位正交方阵
Fpre  = VT.row(8).reshape(0, 3);      // V的最后一列
F矩阵的其秩为2,需要再对Fpre进行奇异值分解, 后取对角矩阵U, 秩为2,后再合成F。
cv::SVDecomp(Fpre,S,U,VT,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
S.at<float>(2)=0;                     //  基础矩阵的秩为2，重要的约束条件
F21 = U * cv::Mat::diag(S)* VT        // 再合成F

```
#### Initializer::CheckFundamental() 计算基本矩阵得分       得分为SF
```asm
思想:
	1. 根据基本矩阵，可以求得两组对点相互变换得到的误差平方和，
	2. 基于卡方检验计算出的阈值（假设测量有一个像素的偏差），
	3. 误差大的记录为外点，误差小的记录为内点，
	4. 使用阈值减去内点的误差，得到该匹配点对的得分(误差小的，得分高)，
	5. 记录相互变换中所有内点的得分之和作为该单元矩阵的得分，并更新匹配点对的内点外点标记。
	
步骤1: 误差阈值 th = 3.841; 得分最大值 thScore = 5.991; 误差方差倒数 invSigmaSquare
步骤2: 遍历每一对2d-2d点对计算误差：

        如： p2 ->  F12  -> p1 
		     |f1   f2   f3|     |u1|
	|u2 v2 1| *  |f4   f5   f6|  *  |v1|    = 0 , 应该=0 不等于零的就是误差
		     |f7   f8   f9|     |1|
前两项展开得到：
	a1 = f1*u2 + f4*v2 + f7;
	b1 = f2*u2 + f5*v2 + f8;
	c1 = f3*u2 + f6*v2 + f9;
得到：
		     |u1|
	|a1 b1 c1| * |v1| = 0 ，其实就是点 (u1, v1) 在线段 (a1, b1, c1) 上的形式
		     |1|
极线l1：a1*x + b1*y + c1 = 0,这里把p2投影到帧1平面上对应的极线形式,p1应该在上面。
点p1,(u1,v1) 到 l1的距离为：    
        d = |a1*u + b1*v + c| / sqrt(a1^2 + b1^2) 
距离平方：
        chiSquare1 = d^2 = (a1*u + b1*v + c)^2 / (a1^2 + b1^2)

根据方差归一化误差:
	const float chiSquare1 = squareDist1*invSigmaSquare;
	
使用阈值更新内外点标记 并记录内点得分:
	if(chiSquare1 > th)
	    bIn = false;                  // 距离大于阈值  该点 变换的效果差，记录为外点
	else
	    score += thScore - chiSquare1;// 得分上限 - 距离差值 得到 得分，差值越小，得分越高      得分为SF

同时记录 p1 ->  F21  -> p2 的误差，也如上述步骤。
更新内外点记录数组：
	if(bIn)
		vbMatchesInliers[i]=true;// 是内点  误差较小
	else
		vbMatchesInliers[i]=false;// 是野点 误差较大
```

### b. homography(单应性矩阵) 随机采样 找到最好的单元矩阵 Initializer::FindHomography()
```asm
思想：
计算单应矩阵,随机采样序列4点法,采用归一化的直接线性变换（normalized DLT）求解，
假设场景为平面情况下通过前两帧求取Homography矩阵(current frame 2 到 reference frame 1)，
在最大迭代次数内调用 ComputeH21计算H,
使用 CheckHomography 计算此单应变换得分，
在最大迭代次数内保留最高得分的单应矩阵H。

步骤：
步骤1：将两帧上对应的2d-2d匹配点对进行归一化
	Initializer::Normalize(mvKeys1,vPn1, T1);//  mvKeys1原坐标 vPn1归一化坐标，T1标准化矩阵
	Initializer::Normalize(mvKeys2,vPn2, T2);// 
	cv::Mat T2inv = T2.inv();                // 标准化矩阵 逆矩阵
步骤2：在最大迭代次数mMaxIterations内，从标准化后的点中随机选取8对点对
	int idx = mvSets[it][j];                 //随机数集合 总匹配点数范围内
	vPn1i[j] = vPn1[mvMatches12[idx].first];
	vPn2i[j] = vPn2[mvMatches12[idx].second]; 
步骤3：通过标准化逆矩阵和标准化点对的单应变换计算原点对的单应变换矩阵 Initializer::ComputeH21()
	cv::Mat Hn = ComputeH21(vPn1i,vPn2i);// 计算 标准化后的点对 对应的 单应矩阵Hn
	// H21i = T2逆 * Hn * T1  见上面  0步骤的推导
	H21i = T2inv * Hn * T1;// 原始点    p1 -----------> p2 的单应
	H12i = H21i.inv();     // 原始点    p2 -----------> p1 的单应
步骤4：通过计算重投影误差来计算单应矩阵的好坏，得分 
	currentScore =Initializer::CheckHomography(); 
步骤5：保留迭代中，得分最高的单应矩阵和对应的得分
	if(currentScore > score)//此次迭代 计算的单应H的得分较高
	{
	    H21 = H21i.clone();//保留较高得分的单应
	    vbMatchesInliers = vbCurrentInliers;//对应的匹配点对   
	    score = currentScore;// 最高的得分
	}
```
#### Initializer::ComputeH21()  4对点直接线性变换求解H矩阵
```asm
一点对：
	p2   =  H21 * p1
写成矩阵形式：
	u2         h1  h2  h3       u1
	v2  =      h4  h5  h6    *  v1
	1          h7  h8  h9       1  

可以使用叉乘 得到0    p2叉乘p2 = H21 *p1 = 0 
	| 0 -1  v2|    |h1 h2 h3|      |u1|    |0|
	| 1  0 -u2| *  |h4 h5 h6| *    |v1| =  |0|
	|-v2 u2  0|    |h7 h8 h9|      |1 |    |0|

也可以展开得到(使用第三项进行归一化)：
	u2 = (h1*u1 + h2*v1 + h3) /( h7*u1 + h8*v1 + h9)
	v2 = (h4*u1 + h5*v1 + h6) /( h7*u1 + h8*v1 + h9)
写成矩阵形式：
	-((h4*u1 + h5*v1 + h6) - ( h7*u1*v2 + h8*v1*v2 + h9*v2))=0  右乘分子，再移动得0
	h1*u1    + h2*v1 + h3  - ( h7*u1*u2 + h8*v1*u2 + h9*u2) =0

	|0    0   0  -u1  -v1  -1   u1*v2   v1*v2    v2|
	|u1 v1    1  0    0    0   -u1*u2  - v1*u2  -u2| *|h1 h2 h3 h4 h5 h6 h7 h8 h9|转置  = 0
一对点提供两个约束：H 9个元素，8个自由度，包含一个比例因子(尺度来源)，需要四对点
四对点提供8个约束方程,可以写成矩阵形式：
 A * h = 0
对A进行SVD奇异值分解 [U,S,V]=svd(A)，其中U和V代表二个相互正交矩阵，而S代表一对角矩阵
cv::SVDecomp(A,S,U,VT,SVD::FULL_UV);  //后面的FULL_UV表示把U和VT补充称单位正交方阵	
H = VT.row(8).reshape(0, 3);// v的最后一列

SVD奇异值分解 解齐次方程组（Ax = 0）原理：
	把问题转化为最小化|| Ax ||2的非线性优化问题，
	我们已经知道了x = 0是该方程组的一个特解，
	为了避免x = 0这种情况（因为在实际的应用中x = 0往往不是我们想要的），
	我们增加一个约束，比如|| x ||2 = 1，
	这样，问题就变为：
	min(|| Ax ||2) ， || x ||2 = 1 或 min(|| Ax ||) ， || x || = 1
	对矩阵A进行分解 A = UDV'
	 || Ax || = || UDV' x || = || DV' x||
	 ( 对于一个正交矩阵U，满足这样一条性质： || UD || = || D || , 
	   正交矩阵，正交变换，仅仅对向量，只产生旋转，无尺度缩放，和变形，即模长不变)

	令y = V'x， 因此，问题变为min(|| Dy ||)， 
	因为|| x || = 1, V'为正交矩阵，则|| y || = 1。
	由于D是一个对角矩阵，对角元的元素按递减的顺序排列，因此最优解在y = (0, 0,..., 1)'
	又因为x = Vy， 所以最优解x，就是V的最小奇异值对应的列向量，
	比如，最小奇异值在第8行8列，那么x = V的第8个列向量。
```
#### 计算单应变换得分 Initializer::CheckHomography()      得分为SH
```asm
思想:
	1. 根据单应变换，可以求得两组对点相互变换得到的误差平方和，
	2. 基于卡方检验计算出的阈值（假设测量有一个像素的偏差），
	3. 误差大的记录为外点，误差小的记录为内点，
	4. 使用阈值减去内点的误差，得到该匹配点对的得分(误差小的，得分高)，
	5. 记录相互变换中所有内点的得分之和作为该单元矩阵的得分，并更新匹配点对的内点外点标记。

步骤1：获取单应变换误差阈值th,以及误差归一化的方差倒数    
	const float th = 5.991;                        // 单应变换误差 阈值
	const float invSigmaSquare = 1.0/(sigma*sigma);//方差 倒数，用于将误差归一化

步骤2：遍历每个点对，计算单应矩阵 变换 时产生 的 对称的转换误差
  p1点 变成 p2点  p2 = H21*p1---------------------------------------
	|u2'|      |h11  h12  h13|     |u1|    |h11*u1 + h12*v1 + h13|
	|v2'|  =   |h21  h22  h23|  *  |v1| =  |h21*u1 + h22*v1 + h23|
	|1|        |h31  h32  h33|     |1|     |h31*u1 + h32*v1 + h33|
	 使用第三行进行归一化： u2' = (h11*u1 + h12*v1 + h13)/(h31*u1 + h32*v1 + h33);
			      v2' = (h21*u1 + h22*v1 + h23)/(h31*u1 + h32*v1 + h33);
	 所以 p1通过 H21 投影到另一帧上 应该和p2的左边一致，但是实际不会一致，可以求得坐标误差
	 误差平方和            squareDist2 = (u2-u2')*(u2-u2') + (v2-v2')*(v2-v2');
	 使用方差倒数进行归一化 chiSquare2 = squareDist2*invSigmaSquare;

  使用阈值更新内外点标记 并记录内点得分:
	if(chiSquare2>th)
		bIn = false;              //距离大于阈值  该点 变换的效果差，记录为外点
	else
		score += th - chiSquare2; // 阈值 - 距离差值 得到 得分，差值越小  得分越高

  p2点 变成 p1点  p1 = H12 * p2 ------------------------------------
	 |u1'|     |h11inv   h12inv   h13inv|    |u2|   |h11inv*u2 + h12inv*v2 + h13inv|
	 |v1'|  =  |h21inv   h22inv   h23inv|  * |v2| = |h21inv*u2 + h22inv*v2 + h23inv|
	 |1|       |h31inv   h32inv   h33inv|    |1|    |h31inv*u2 + h32inv*v2 + h33inv|
	 使用第三行进行归一化： u1' = (h11inv*u2 + h12inv*v2 + h13inv)/(h31inv*u2 + h32inv*v2 + h33inv);
			      v1' = (h21inv*u2 + h22inv*v2 + h23inv)/(h31inv*u2 + h32inv*v2 + h33inv);
	 所以 p1通过 H21 投影到另一帧上 应该和p2的左边一致，但是实际不会一致，可以求得坐标误差
	 误差平方和            squareDist1 = (u1-u1')*(u1-u1') + (v1-v1')*(v1-v1');
	 使用方差倒数进行归一化 chiSquare1 = squareDist1*invSigmaSquare; 

  使用阈值更新内外点标记 并记录内点得分:
	if(chiSquare1>th)    
		bIn = false;             // 距离大于阈值  该点 变换的效果差，记录为外点
	else
		score += th - chiSquare1;// 阈值 - 距离差值 得到 得分，差值越小  得分越高  SH

  更新内外点记录数组：
	if(bIn)
		vbMatchesInliers[i]=true;// 是内点  误差较小
	else
		vbMatchesInliers[i]=false;// 是野点 误差较大
```

#### 从两个模型 H F 得分为 Sh   Sf 中选着一个 最优秀的 模型 的方法为
	文中认为，当场景是一个平面、或近似为一个平面、或者视差较小的时候，可以使用单应性矩阵H恢复运动，
	当场景是一个非平面、视差大的场景时，使用基础矩阵F恢复运动,
	两个变换矩阵得分分别为 SH 、SF,
        根据两者得分计算一个评价指标RH:
	RH = SH / ( SH + SF)
	当大于0.45时，选择从单应性变换矩阵还原运动,反之使用基础矩阵恢复运动。
	不过ORB_SLAM2源代码中使用的是0.4作为阈值。

### c. 单应矩阵H恢复R,t  Initializer::ReconstructH()
	单应矩阵恢复  旋转矩阵 R 和平移向量t
	p2   =  H21 * p1   
	p2 = K( RP + t)  = KTP = H21 * KP  
	A = T =  K 逆 * H21*K = [ R t; 0 0 0 1]
	对A进行奇异值分解
	cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
	使用FAUGERAS的论文[1]的方法，提出8种运动假设,分别可以得到8组R,t
	正常来说，第二帧能够看到的点都在相机的前方，即深度值Z大于0.
	但是如果在低视差的情况下，使用 三角变换Initializer::Triangulate() 得到的3d点云，
	使用 Initializer::CheckRT() 统计符合该R，t的内点数量。
#### Initializer::Triangulate() 使用2d-2d点对 和 变换矩阵 R，t 三角变换恢复2d点对应的3d点
	Trianularization: 已知匹配特征点对{p1  p2} 和 
	各自相机投影矩阵{P1 P2}, P1 = K*[I 0], P2 = K*[R t], 尺寸为[3,4]
	估计三维点 X3D
	     p1 = P1 * X3D
	     p2 = P2 * X3D
	采用直接线性变换DLT的方法(将式子变换成A*X=0的形式后使用SVD奇异值分解求解线性方程组)：
	对于 p1 = P1 * X3D: 方程两边 左边叉乘 p1，可使得式子为0
	p1叉乘P1*X3D = 0
	其叉乘矩阵为:
	叉乘矩阵 =  
		  |0  -1  y|
		  |1   0 -x| 
		  |-y  x  0| 
	上述等式可写: 
	  |0  -1  y|  |P1.row(0)|  
	  |1   0 -x| *|P1.row(1)|* X3D = 0
	  |-y  x  0|  |P1.row(2)|  

	对于第一行 |0  -1  y| 会与P的三行分别相乘 得到四个值 与齐次3d点坐标相乘得到 0
	有 (y * P1.row(2) - P1.row(1) ) * X3D = 0
	对于第二行 |1   0 -x|有：
	有 (x * P1.row(2) - P1.row(0) ) * X3D = 0
	得到两个约束，另外一个点 p2 = P2 * X3D,也可以得到两个式子：
	(y‘ * P2.row(2) - P2.row(1) ) * X3D = 0
	(x’ * P2.row(2) - P2.row(0) ) * X3D = 0
	写成 A*X = 0的形式有：
	A =(维度4*4)
	|y * P1.row(2) - P1.row(1) |
	|x * P1.row(2) - P1.row(0) |
	|y‘ * P2.row(2) - P2.row(1)|
	|x’ * P2.row(2) - P2.row(0)|
	对A进行奇异值分解求解X
	cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
	x3D = vt.row(3).t();// vt的最后一列,为X的解
	x3D = x3D.rowRange(0,3)/x3D.at<float>(3);//  转换成非齐次坐标  归一化

#### initializer::CheckRT() 计算R , t 的得分(内点数量)
	统计规则，使用 Initializer::Triangulate()  三角化的3d点，
	会有部分点云跑到相机的后面(Z<0)，该部分点是噪点，需要剔除.
	另外一个准则是将上述符合条件的3d点分别重投影会前后两帧上，
	与之前的orb特征点对的坐标做差，误差平方超过阈值，剔除。
	统计剩余符合条件的点对数量。
	在8种假设中，记录得到最大内点数量 和 次大内点数量 
	当最大内点数量 远大于 次大内点数量( 有明显的差异性)，
	该最大内点数量对应的R,t，具有较大的可靠性，返回该，R，t。


### d. 基础矩阵F恢复R,t  Initializer::ReconstructF()
	计算 本质矩阵 E  =  K转置 * F  * K
	E = t叉乘 R
	奇异值分解 E= U C  V   , U 、V 为正交矩阵， C 为奇异值矩阵  C =  diag(1, 1, 0)
	从本质矩阵E 恢复 旋转矩阵R 和 平移向量t
	有四种假设 得到四种 R,t
	理论参考 Result 9.19 in Multiple View Geometry in Computer Vision
	使用 initializer::CheckRT() 计算R , t 的得分(内点数量)
	使用4中情况中得分最高的R,t 作为最终恢复的R，t
 
### e. 单目创建初始地图      CreateInitialMapMonocular();
	思想：
		使用单目相机前两帧生成的3d点创建初始地图，
		使用全局地图优化 Optimizer::GlobalBundleAdjustemnt(mpMap,20)优化地图点，
		计算场景的深度中值(地图点深度的中位数)，
		使用平均逆深度归一化 两帧的平移变换量t,
		使用 平均逆深度归一化地图点。
	步骤：
	步骤1：使用前两帧创建关键帧，并把这 两帧 关键帧 加入地图，加入关键帧数据库。
	步骤2：使用之前三角化得到的3d点创建 地图点。
	步骤3：地图点和每一帧的2d点对应起来，在关键帧中添加这种对应关系。 关键帧关联地图点。
	步骤4：关键帧和2d点对应起来，在地图点中添加这种关系。            地图点关联关键帧。
	步骤5：一个地图点会被许多个关键帧观测到，那么就会关联到许多个2d点，需要更新地图点对应2d点的orb特征描述子。
	步骤6：当前帧关联地图点，地图点加入到地图。
	步骤7：更新关键帧的 连接关系，被地图点观测到的次数。
	步骤8：全局优化地图 BA最小化重投影误差，这两帧姿态进行全局优化。
	       Optimizer::GlobalBundleAdjustemnt(mpMap,20);
	       // 注意这里使用的是全局优化，和回环检测调整后的大回环优化使用的是同一个函数。
	       // 放在后面再进行解析
	步骤9：计算场景深度中值，以及逆深度中值，对位姿的平移量进行尺度归一化，对地图点进行尺度归一化。


## 2. 双目/RGBD初始化 Tracking::StereoInitialization()根据视差计算深度（深度相机直接获取深度）计算3d点，创建地图点，创建初始地图
	步骤：
		当前帧 特征点个数 大于500 进行初始化
		设置第一帧为关键帧，并设置为位姿为 T = [I 0]，世界坐标系 
		创建关键帧，地图添加关键帧，
		根据每一个2d点的视差（左右两张图像，orb特征点金字塔分层分块匹配得到视差d）求得的深度D=fB/d，计算对应的3D点，并创建地图点，
		地图点关联观测帧 地图点计算所有关键帧中最好的描述子并更新地图点的方向和距离，
		关键帧关联地图点，地当前帧添加地图点  地图添加地图点
		局部地图中添加该初始关键帧。

# 4.3 两帧跟踪得到初始化位姿(跟踪上一帧/跟踪参考帧/重定位)
[参考](https://blog.csdn.net/u010128736/article/details/53339311)
### a. Tracking::TrackWithMotionModel() 跟踪上一帧模式， 相机移动量较大，当前帧和上一帧相差较大，做匹配三角变换可以获得很好的效果



### b. Tracking::TrackReferenceKeyFrame() 跟踪参考帧模式，相机移动量较小，和上一帧相差不大，需要和前面的帧(参考帧)匹配

### c. bool Tracking::Relocalization() 上面两种模式都没有跟踪成功，需要使用重定位模式，使用orb字典编码在关键帧数据库中找相似的关键帧，进行匹配跟踪

			 
# 4.4 跟踪局部地图，多帧局部地图G2O优化位姿, 新建关键帧
[参考](https://blog.csdn.net/u010128736/article/details/53395936)
上面完成初始位姿的跟踪后，需要使用局部地图类进行局部地图优化，来提高鲁棒性
局部地图中与当前帧有相同点的关键帧序列成为一级相关帧K1，
而与一级相关帧K1有共视地图点的关键帧序列成为二级相关帧K2，在其中搜索局部地图点，投影到当前帧上，
使用 位姿优化 Optimizer::PoseOptimization(&mCurrentFrame)， 进行优化，
更新 地图点的信息(关键帧的观测关系)



# 4.5 闭环检测线程 LoopClosing
[参考](https://blog.csdn.net/u010128736/article/details/53409199)

 
# 5. 数学理论总结
[参考]()


