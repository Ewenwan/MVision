    ##########################################################
# F 点云配准 Registration 
 
[4点法粗采样](http://graphics.stanford.edu/~niloy/research/fpcs/paper_docs/fpcs_slides_sig_08.pdf)

[Super4PCS: Fast Global Pointcloud Registration via Smart Indexing](http://geometry.cs.ucl.ac.uk/projects/2014/super4PCS/)

[开源ICP库，trimesh2，实测下来应该是开源ICP算法中效果最好的，谁用谁知道 ](https://github.com/Ewenwan/trimesh2)

    #################################################

    在逆向工程，计算机视觉，文物数字化等领域中，由于点云的不完整，旋转错位，平移错位等，
    使得要得到的完整的点云就需要对局部点云进行配准，为了得到被测物体的完整数据模型，
    需要确定一个合适的坐标系，将从各个视角得到的点集合并到统一的坐标系下形成一个完整的点云，
    然后就可以方便进行可视化的操作，这就是点云数据的配准。

    实质就是把不同的坐标系中测得到的数据点云进行坐标系的变换，以得到整体的数据模型，
    问题的关键是如何让得到坐标变换的参数R（旋转矩阵）和T（平移向量），
    使得两视角下测得的三维数据经坐标变换后的距离最小，

    目前配准算法按照过程可以分为整体配准和局部配准。
    PCL中有单独的配准模块，实现了配准相关的基础数据结构，和经典的配准算法如ICP。 

    给定两个来自不同坐标系的三维数据点集，找到两个点集空间的变换关系，
    使得两个点集能统一到同一坐标系统中，即配准过程
    
    求得旋转和平移矩阵
    P2 = R*P1  + T　　　　[R t]
    
    点云配准的概念也可以类比于二维图像中的配准，
    
    只不过二维图像配准获取得到的是x，y，alpha，beta等放射变化参数

    三维点云配准可以模拟三维点云的移动和对其，也就是会获得一个旋转矩阵和一个平移向量，
    通常表达为一个4×3的矩阵，其中3×3是旋转矩阵，1*3是平移向量。
    严格说来是6个参数，因为旋转矩阵也可以通过罗格里德斯变换转变成1*3的旋转向量。

## 常用的点云配准算法有两种：
        1. 正态分布变换方法  NDT  正态分布变换进行配准（normal Distributions Transform） 
        2. 和著名的 迭代最近点 Iterative Closest Point （ICP） 点云配准，
                    
## 此外还有许多其它算法  列举如下：
        ICP：稳健ICP、point to plane ICP、point to line ICP、MBICP、GICP
        NDT: NDT 3D、Multil-Layer NDT
        FPCS、KFPSC、SAC-IA
        Line Segment Matching、ICL

## 两个数据集的计算步骤：
      1. 识别最能代表两个数据集中的场景的兴趣点（interest points）（即关键点 keypoints）
      2. 在每个关键点处，计算特征描述符;
      3. 从特征描述符集合以及它们在两个数据集中的x,y,z位置，基于特征和位置之间的相似性来估计对应关系;
      4. 假设数据被认为包含噪声的，并不是所有的对应关系都是有效的，
         所以舍弃对配准过程产生负面影响的那些负影响对应关系;
      5. 利用剩余的正确的对应关系来估算刚体变换，完整配准。

##  迭代最近点 Iterative Closest Point （ICP）
    ICP算法本质上是基于最小二乘法的最优配准方法。
    该算法重复进行选择对应关系点对，计算最优刚体变换这一过程，直到满足正确配准的收敛精度要求。
    算法的输入：参考点云和目标点云，停止迭代的标准。
    算法的输出：旋转和平移矩阵，即转换矩阵。

    使用点匹配时，使用点的XYZ的坐标作为特征值，针对有序点云和无序点云数据的不同的处理策略：
          1. 穷举配准（brute force matching）;
          2. kd树最近邻查询（FLANN）;
          3. 在有序点云数据的图像空间中查找;
          4. 在无序点云数据的索引空间中查找.
    特征描述符匹配：
          1. 穷举配准（brute force matching）;
          2. kd树最近邻查询（FLANN）。

    in cloud B for every point in cloud A .

## 错误对应关系的去除（correspondence rejection）:

        由于噪声的影响，通常并不是所有估计的对应关系都是正确的，
        由于错误的对应关系对于最终的刚体变换矩阵的估算会产生负面的影响，
        所以必须去除它们，可以采用随机采样一致性估计，或者其他方法剔除错误的对应关系，
        最终只使用一定比例的对应关系，这样既能提高变换矩阵的估计京都也可以提高配准点的速度。

## 变换矩阵的估算（transormation estimation）的步骤如下:

          1. 在对应关系的基础上评估一些错误的度量标准
          2. 在摄像机位姿（运动估算）和最小化错误度量标准下估算一个刚体变换(  rigid  transformation )
          3. 优化点的结构  (SVD奇异值分解 运动估计;使用Levenberg-Marquardt 优化 运动估计;)
          4. 使用刚体变换把源旋转/平移到与目标所在的同一坐标系下，用所有点，点的一个子集或者关键点运算一个内部的ICP循环
          5. 进行迭代，直到符合收敛性判断标准为止。


    ===================================================

## 迭代最近点算法（Iterative CLosest Point简称ICP算法）:
    ICP算法对待拼接的2片点云，首先根据一定的准则确立对应点集P与Q，
    其中对应点对的个数，然后通过最小二乘法迭代计算最优的坐标变换，
    即旋转矩阵R和平移矢量t，使得误差函数最小，
    
### ICP处理流程分为四个主要的步骤：

        1. 对原始点云数据进行采样(关键点 keypoints(NARF, SIFT 、FAST、均匀采样 UniformSampling)、
           特征描述符　descriptions，NARF、 FPFH、BRIEF 、SIFT、ORB )
        2. 确定初始对应点集(匹配 matching )
        3. 去除错误对应点对(随机采样一致性估计 RANSAC )
        4. 坐标变换的求解

### Feature based registration 配准
        1. SIFT 关键点 (pcl::SIFT…something)
        2. FPFH 特征描述符  (pcl::FPFHEstimation)  
        3. 估计对应关系 (pcl::CorrespondenceEstimation)
        4. 错误对应关系的去除( pcl::CorrespondenceRejectionXXX )  
        5. 坐标变换的求解

###  PCL类的相关的介绍:
    对应关系基类　　   pcl::CorrespondenceGrouping< PointModelT, PointSceneT >
    几何相似性对应　   pcl::GeometricConsistencyGrouping< PointModelT, PointSceneT >
    相似性度量　　　   pcl::recognition::HoughSpace3D
    多实例对应关系　   pcl::Hough3DGrouping< PointModelT, PointSceneT, PointModelRfT, PointSceneRfT >
    CRH直方图　　　   pcl::CRHAlignment< PointT, nbins_ >
    随机采样一致性估计 pcl::recognition::ObjRecRANSAC::Output
          pcl::recognition::ObjRecRANSAC::OrientedPointPair
          pcl::recognition::ObjRecRANSAC::HypothesisCreator
          pcl::recognition::ObjRecRANSAC

          pcl::recognition::ORROctree::Node::Data
          pcl::recognition::ORROctree::Node
          pcl::recognition::ORROctree
          pcl::recognition::RotationSpace
          
### 1. ICP迭代最近点算法  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    ===========================================================
      pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
         //创建IterativeClosestPoint的对象
      icp.setInputCloud(cloud_in);                 //cloud_in设置为点云的源点
      icp.setInputTarget(cloud_out);               //cloud_out设置为与cloud_in对应的匹配目标
      pcl::PointCloud<pcl::PointXYZ> Final;        //存储经过配准变换点云后的点云
      icp.align(Final);  
    结果：
      icp.hasConverged()
      icp.getFitnessScore()
      icp.getFinalTransformation()
    ===========================================================
    
[ICP迭代最近点算法 IterativeClosestPoint ](Basic/Registration/iterative_closest_point.cpp)   
    
    
### 2. 非线性ICP 配准对象  逐步匹配多幅点云  pcl::IterativeClosestPointNonLinear
    pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;   // 非线性ICP 配准对象

    逐步匹配多幅点云
    本实例是使用迭代最近点算法，逐步实现地对一系列点云进行两两匹配，
    他的思想是对所有的点云进行变换，使得都与第一个点云统一坐标系 ，
    在每个连贯的有重叠的点云之间找出最佳的变换，并积累这些变换到全部的点云，
    能够进行ICP算法的点云需要粗略的预匹配(比如在一个机器人的量距内或者在地图的框架内)，
    并且一个点云与另一个点云需要有重叠的部分。

     如果观察不到结果，就按键R来重设摄像头，
    调整角度可以观察到有红绿两组点云显示在窗口的左边，
    红色为源点云，将看到上面的类似结果，命令行提示需要执行配准按下Q，
    按下后可以发现左边的窗口不断的调整点云，其实是配准过程中的迭代中间结果的输出，
    在迭代次数小于设定的次数之前，右边会不断刷新最新的配准结果，
    直到收敛，迭代次数30次完成整个匹配的过程，再次按下Q后会看到存储的1.pcd文件，
    此文件为第一个和第二个点云配准后与第一个输入点云在同一个坐标系下的点云。
[数据](https://github.com/PointCloudLibrary/data/tree/master/tutorials/pairwise)

        // 配准
        pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;   // 配准对象
        reg.setTransformationEpsilon (1e-6);   ///设置收敛判断条件，越小精度越大，收敛也越慢 
        // Set the maximum distance between two correspondences (src<->tgt) to 10cm大于此值的点对不考虑
        // Note: adjust this based on the size of your datasets
        reg.setMaxCorrespondenceDistance (0.1);// 10cm大于此值的点对不考虑
        // 设置点表示
        reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));

        reg.setInputSource (points_with_normals_src);   // 设置源点云
        reg.setInputTarget (points_with_normals_tgt);   // 设置目标点云
        Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;// Ti Source to target
        PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
        reg.setMaximumIterations (2);////设置最大的迭代次数，即每迭代两次就认为收敛，停止内部迭代
        for (int i = 0; i < 30; ++i)   ////手动迭代，每手动迭代一次，在配准结果视口对迭代的最新结果进行刷新显示
        {
        PCL_INFO ("Iteration Nr. %d.\n", i);
        // 存储点云以便可视化
        points_with_normals_src = reg_result;
        // Estimate
        reg.setInputSource (points_with_normals_src);
        reg.align (*reg_result);
        //accumulate transformation between each Iteration
        Ti = reg.getFinalTransformation () * Ti;// keep track of and accumulate the transformations 
            //if the difference between this transformation and the previous one
            //is smaller than the threshold, refine the process by reducing
            //the maximal correspondence distance
        if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
          reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001); 
        prev = reg.getLastIncrementalTransformation ();//　
        // visualize current state  vp_2 右边显示配准 
        showCloudsRight(points_with_normals_tgt, points_with_normals_src);
        }
        // Get the transformation from target to source
        targetToSource = Ti.inverse();//deidao

        // Transform target back in source frame
        pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);
        
[非线性ICP 配准对象  逐步匹配多幅点云 IterativeClosestPointNonLinea ](Basic/Registration/pairwise_incremental_registration.cpp)
 
## 3. 交互式ICP可视化的程序。
        该程序将加载点云并对其进行刚性变换。
        之后，使用ICP算法将变换后的点云与原来的点云对齐。
        每次用户按下“空格”，进行ICP迭代，刷新可视化界面。

        在这里原始例程使用的是PLY格式的文件，可以找一个PLY格式的文件进行实验，
        也可以使用格式转换文件 把PCD 文件转为PLY文件

        Creating a mesh with Blender

        1.  Install and open Blender then 
            delete the cube in the scene by pressing “Del” key :
        2. Add a monkey mesh in the scene :

        3. Subdivide the original mesh to make it more dense :
           Configure the subdivision to 2 or 3 for example : 
           don’t forget to apply the modifier

        4. Export the mesh into a PLY file :
 
[交互式ICP可视化的程序](Basic/Registration/interactive_icp.cpp)

    ==============================================

## 4. 正态分布变换进行配准（normal Distributions Transform）

    介绍关于如何使用正态分布算法来确定两个大型点云之间的刚体变换，
    正态分布变换算法是一个配准算法，它应用于三维点的统计模型，
    使用标准最优化技术来确定两个点云间的最优匹配，
    因为其在配准的过程中不利用对应点的特征计算和匹配，
    所以时间比其他方法比较快.
### code
        // 初始化正态分布(NDT)对象
        pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

        // 根据输入数据的尺度设置NDT相关参数
        ndt.setTransformationEpsilon (0.01);// 为终止条件设置最小转换差异
        ndt.setStepSize (0.1);              // 为more-thuente线搜索设置最大步长
        ndt.setResolution (1.0);            // 设置NDT网格网格结构的分辨率（voxelgridcovariance）

        //以上参数在使用房间尺寸比例下运算比较好，但是如果需要处理例如一个咖啡杯子的扫描之类更小的物体，需要对参数进行很大程度的缩小

        //设置匹配迭代的最大次数，这个参数控制程序运行的最大迭代次数，一般来说这个限制值之前优化程序会在epsilon变换阀值下终止
        //添加最大迭代次数限制能够增加程序的鲁棒性阻止了它在错误的方向上运行时间过长
        ndt.setMaximumIterations (35);

        ndt.setInputSource (filtered_cloud);  //源点云
        // Setting point cloud to be aligned to.
        ndt.setInputTarget (target_cloud);  //目标点云

        // 设置使用机器人测距法得到的粗略初始变换矩阵结果
        Eigen::AngleAxisf init_rotation (0.6931, Eigen::Vector3f::UnitZ ());
        Eigen::Translation3f init_translation (1.79387, 0.720047, 0);
        Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();

        // 计算需要的刚体变换以便将输入的源点云匹配到目标点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        ndt.align (*output_cloud, init_guess);
    
[download](http://pointclouds.org/documentation/tutorials/normal_distributions_transform.php#normal-distributions-transform)
[正态分布变换进行配准 ](Basic/Registration/normal_distributions_transform.cpp)
    ===================================

## 物体三维模型重建　3d scanner for small objects

### 1. 输入数据处理：
      计算法线、分割
           通过将阈值应用到HSV颜色空间中的颜色中，将前景点分割为手和目标区域。
          手部区域扩大了几个像素，以减少意外将手点包括到物体云中的危险。
### 2. 点云配准:
      使用迭代最近点（ICP）算法将处理过的数据点云与普通模型网格对齐。
      适应度：　拒绝后对应关系的均方欧氏距离。
      预选择：　丢弃面向传感器的模型点。
      对应估计：利用KD树进行最近邻搜索。
      对应的排斥：
        丢弃与高于阈值的平方欧几里德距离的对应关系。
        阈值以无穷大初始化（第一次迭代中没有拒绝），
        并将最后一次迭代的适应度乘以用户定义的因子。
        丢弃它们的法线之间的夹角高于用户定义的阈值的对应关系。
      变换估计：以数据云为源，以模型网格为目标的点到面距离最小化。
      收敛准则：
        Epsilon：当当前和前一个迭代之间的适应度变化小于用户定义的ε值时，就会检测到收敛。
      破坏准则：　超过最大迭代次数。
      适应度大于用户定义的阈值（在收敛状态下进行评估）。
      模型网格和数据云之间的重叠小于用户定义的阈值（在收敛状态下进行评估）。

### 3. 模型合并:
      重构初始模型网格（无组织），并将已注册的数据云（组织）与模型合并。

      合并是通过从数据云搜索最近的邻居到模型网格，
      如果它们的法线之间的夹角小于给定的阈值，则平均算出相应的点。
      如果平方欧氏距离高于给定的平方距离阈值，则将数据点添加到网格中作为新的顶点。
      数据云的组织性质被用来连接人脸。

      离群值剔除是基于离群值不能从多个不同方向观察的假设。
      因此，每个顶点存储一个可见性信任，它是记录它的唯一方向的数量。
      顶点得到一定的时间（最大年龄），直到他们必须达到最小可见性信心，然后再从网格中移除。
      顶点存储一个由零初始化并在每次迭代中增加的年龄。
      如果顶点在当前合并步骤中有对应关系，则将年龄重置为零。
      此设置确保当前正在合并的顶点始终保持在网格中，而不考虑可见性。
      一旦物体转过身，某些顶点就看不见了。
      年龄的增加，直到他们达到最大年龄时，如果他们被保留在网格或删除。
### code 
        // 估计物体点云　和　场景点云　fphf特征　Estimate features
        pcl::console::print_highlight ("Estimating features...\n");
        FeatureEstimationT fest;//　fphf特征估计
        fest.setRadiusSearch (0.025);//　搜索半径
        fest.setInputCloud (object);
        fest.setInputNormals (object);
        fest.compute (*object_features);//　物体点云　fphf特征
        fest.setInputCloud (scene);
        fest.setInputNormals (scene);
        fest.compute (*scene_features);//　场景点云　fphf特征

        // SampleConsensusPrerejective随机采样一致性　配准　Perform alignment
        pcl::console::print_highlight ("Starting alignment...\n");
        pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;//配准
        align.setInputSource (object);//源点云
        align.setSourceFeatures (object_features);//源点云　fphf特征
        align.setInputTarget (scene);//目标点云
        align.setTargetFeatures (scene_features);//目标点云　　特征
        align.setMaximumIterations (50000); // 　RANSAC 　最大迭代次数
        align.setNumberOfSamples (3); // 采样点数　Number of points to sample for generating/prerejecting a pose
        align.setCorrespondenceRandomness (5);// 使用的特征数量　Number of nearest features to use
        align.setSimilarityThreshold (0.9f); // 相似性　阈值　Polygonal edge length similarity threshold
        align.setMaxCorrespondenceDistance (2.5f * leaf);// 内点　阈值　Inlier threshold
        align.setInlierFraction (0.25f); // Required inlier fraction for accepting a pose hypothesis
        {
        pcl::ScopeTime t("Alignment");
        align.align (*object_aligned);
        }
  
[物体三维模型重建 ](Basic/Registration/alignment_prerejective.cpp)
