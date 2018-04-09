# 点云识别
## 1 基于对应分组的三维物体识别

        基于pcl_recognition模块的三维物体识别。
        具体来说，它解释了如何使用对应分组算法，
        以便将3D描述符匹配阶段之后获得的一组点到点对应集集合到当前场景中的模型实例中。
        每个集群，代表一个可能的场景中的模型实例，
        对应的分组算法输出的
        变换矩阵识别当前的场景中，
        模型的六自由度位姿估计 6DOF pose estimation 。
        
执行命令
	./correspondence_grouping ../milk.pcd ../milk_cartoon_all_small_clorox.pcd -c -k

	输入一个具体的物体的点云，从场景中找出与该物体点云相匹配的，这种方法可以用来抓取指定的物体等等.


	【1】计算法线向量 pcl::NormalEstimationOMP 
	     近邻邻域内 协方差矩阵PCA 降维到二维平面 计算法线向量
	   [PCA降维原理](http://blog.codinglabs.org/articles/pca-tutorial.html)
	   前面说的是二维降到一维时的情况，假如我们有一堆散乱的三维点云,则可以这样计算法线：
	   1）对每一个点，取临近点，比如取最临近的50个点，当然会用到K-D树
	   2）对临近点做PCA降维，把它降到二维平面上,可以想象得到这个平面一定是它的切平面(在切平面上才可以尽可能分散）
	   3）切平面的法线就是该点的法线了，而这样的法线有两个，取哪个还需要考虑临近点的凸包方向

	【2】下采样滤波使用均匀采样 pcl::UniformSampling
            （可以试试体素格子下采样）得到关键点
	    一般下采样是通过构造一个三维体素栅格（正方体格子内保留一个点），
	    然后在每个体素内用体素内的所有点的重心近似显示体素中的其他点，
	    这样体素内所有点就用一个重心点来表示，进行下采样的来达到滤波的效果，
	    这样就大大的减少了数据量，
	    特别是在配准，曲面重建等工作之前作为预处理，
	    可以很好的提高程序的运行速度，
	    均匀采样 (半径球体内保留一个点)

	【3】为keypoints关键点计算SHOT(法线方向直方图特征)描述子 pcl::SHOTEstimationOMP
　　　　　　　	     1. 半径r内划分32个空间区域
	　　   2. 计算区域内每个点与中心点p法线np之间的夹角余弦cosθ=nv·np
　　　　　　　      3. 对夹角余弦值进行11个格子直方图统计
	　　   4. 计算结果进行归一化得到一个352（32*11=352）维特征

	【4】按存储方法KDTree匹配两个点云（描述子向量匹配）点云分组得到匹配的组 描述 点对匹配关系
	     1. pcl::KdTreeFLANN<DescriptorType> 描述最距离最近临近
	　　　　　2. 为每一个场景点云　在　模型点云中匹配一个最近点　（<0.25f 为匹配）
	【5】参考帧 霍夫聚类 / 集合一致性 聚类得到 匹配点云cluster  变换矩阵和 匹配点对关系
　　　　　　　估计模型参考帧 pcl::BOARDLocalReferenceFrameEstimation  
               霍夫聚类  pcl::Hough3DGrouping
             几何一致性性质   pcl::GeometricConsistencyGrouping
	     
	【6】分组显示 平移矩阵 T 将模型点云按T变换后显示 以及显示 点对之间的连线

## 1.1 shot 特征描述
	构造方法：
	以查询点p为中心构造半径为r 的球形区域，沿径向、方位、俯仰3个方向划分网格，
	其中径向2次，方位8次（为简便图中径向只划分了4个），俯仰2次划分网格，
	将球形区域划分成32(2*8*2=32)个空间区域。
	在每个空间区域计算计算落入该区域点的法线nv和中心点p法线np之间的夹角余弦cosθ=nv·np，
	再根据计算的余弦值对落入每一个空间区域的点数进行直方图统计（划分11个），
	对计算结果进行归一化，使得对点云密度具有鲁棒性，得到一个352维特征（32*11=352）。
	（原论文：Unique Signatures of Histograms for Local Surface）
[基于对应分组的三维物体识别](Recognition/correspondence_grouping.cpp)      
        
        
# 2  隐式形状模型 ISM （Implicit Shape Model） 训练模型　识别模型点云
        隐式形状模型 ISM （隐形状模型 （Implicit Shape Model））
        原理类似视觉词袋模型
        计算所有　训练数据　点云的　特征点和特征描述子　ｋ均值聚类　得到视觉词典

                这个算法是把Hough转换和特征近似包进行结合。
                有训练集，这个算法将计算一个确定的模型用来预测一个物体的中心。

        ISM算法是 《Robust Object Detection with Interleaved Categorization and Segmentation》正式提出的。
        大牛作者叫Bastian Leibe，他还写过其它几篇关于ISM算法的文章。
        该算法用于行人和车的识别效果良好。
[主页](http://www.vision.rwth-aachen.de/software/ism)

        Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes
        用于街景语义分割的全分辨率残差网络（CVPR-12）


        这个算法由两部分组成，第一部分是训练，第二部分是物体识别。
        训练，它有以下6步:
                1.检测关键点，keypoint detection。这只是一个训练点云的简化。
                        在这个步骤里面所有的点云都将被简化，通过体素格下采样　voxel grid　这个途径。
                        余下来的点就是特征点；
                2.对特征点，计算快速点特征直方图特征　FPFH，需要计算法线特征；
                3.通过k-means聚类算法对特征进行聚类得到视觉（几何）单词词典；
                4.计算每一个实例（聚类簇，一个视觉单词）里面的特征关键点　到　聚类中心关键点　的　方向向量；
                5.对每一个视觉单词，依据每个关键点和中心的方向向量，计算其统计学权重；

                6.对每一个关键点计算学习权重，与关键点到聚类中心距离有关。

        我们在训练的过程结束以后，接下来就是对象搜索的进程。
                1.特征点检测。
                2.对每个特征点计算　特征描述子。
                3.对于每个特征点对应的特征描述子搜索最近的　训练阶段得到的视觉单词。
                4.对于每一个特征点计算　类别投票权重（视觉单词统计学权重　关键点学习权重）。
                5.前面的步骤给了我们一个方向集用来预测中心与能量。

        上面的步骤很多涉及机器学习之类的，大致明白那个过程即可.

        ./implicit_shape_model
              ism_train_cat.pcd      0
              ism_train_horse.pcd    1
              ism_train_lioness.pcd  2
              ism_train_michael.pcd  3
              ism_train_wolf.pcd     4
              ism_test_cat.pcd       0
[数据](http://pointclouds.org/documentation/tutorials/implicit_shape_model.php#implicit-shape-model)
[隐式形状模型 ISM ](Recognition/implicit_shape_model.cpp)

# 3 3D物体识别的假设检验 对应分组的三维物体识别 icp点云配准　验证结果
        3D物体识别的假设检验 
        如何做3D物体识别通过验证模型假设在聚类里面。在描述器匹配后，
        这次我们将运行某个相关组算法在PCL里面为了聚类点对点相关性的集合，
        决定假设物体在场景里面的实例。
        在这个假定里面，全局假设验证算法将被用来减少错误的数量。


[对应分组的三维物体识别 icp点云配准　验证结果](Recognition/hypothesis_Ver_Object_Rec.cpp)

