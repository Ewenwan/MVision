/**
* This file is part of ORB-SLAM2.
* 全局/局部 优化 使用G2O图优化
* 
* http://www.cnblogs.com/luyb/p/5447497.html
* 
* 优化的目标函数在SLAM问题中，常见的几种约束条件为：
* 1. 三维点到二维特征的映射关系（通过投影矩阵）；
* 2. 位姿和位姿之间的变换关系（通过三维刚体变换）；
* 3. 二维特征到二维特征的匹配关系（通过F矩阵）；
* 5. 其它关系（比如单目中有相似变换关系）。
* 如果我们能够知道其中的某些关系是准确的，
* 那么可以在g2o中定义这样的关系及其对应的残差，
* 通过不断迭代优化位姿来逐步减小残差和，从而达到优化位姿的目标。
* 
*
1 局部优化
当新的关键帧加入到convisibility graph时，作者在关键帧附近进行一次局部优化，如下图所示。
Pos3是新加入的关键帧，其初始估计位姿已经得到。此时，Pos2是和Pos3相连的关键帧，
X2是Pos3看到的三维点，X1是Pos2看到的三维点，这些都属于局部信息，
共同参与Bundle Adjustment。同时，Pos1也可以看到X1，但它和Pos3没有直接的联系，
属于Pos3关联的局部信息，
参与Bundle Adjustment，但Pos1取值保持不变（位姿固定）。
Pos0和X0不参与Bundle Adjustment。

2全局优化
在全局优化中，所有的关键帧（除了第一帧 Pos0 （位姿固定））和三维点都参与优化


3闭环处的Sim3位姿优化
当检测到闭环时，闭环连接的两个关键帧的位姿需要通过Sim3优化（以使得其尺度一致）。
优化求解两帧之间的 相似变换矩阵 S12，使得二维对应点（feature）的投影误差最小。
如下图所示，Pos6和Pos2为一个可能的闭环。通过(u4,2,v4,2)
和(u4,6,v4,6)之间的投影误差来优化S6,2。

*/

#include "Optimizer.h"
// 函数优化方法
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
//优化方法 莱文贝格－马夸特方法（Levenberg–Marquardt algorithm）能提供数非线性最小化（局部最小）的数值解。
// 矩阵 分解 求解器
#include "Thirdparty/g2o/g2o/core/block_solver.h"//矩阵快分解 求解器的实现。主要来自choldmod, csparse。在使用g2o时要先选择其中一种。
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"// 矩阵 线性优化求解器
// #include <g2o/solvers/csparse/linear_solver_csparse.h>  // csparse求解器
// #include <g2o/solvers/dense/linear_solver_cholmod.h //
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"// 稠密 矩阵 线性求解器
// 图 边顶点的类型
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"// 定义好的顶点类型  6维度 优化变量  例如 相机 位姿
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"// 定义好的顶点类型  7维度 优化变量  例如 相机 位姿 + 深度信息
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include<Eigen/StdVector>
#include "Converter.h"
#include<mutex>

namespace ORB_SLAM2
{

    // 全局优化  地图  迭代次数
/**
 * @brief    pMap中所有的MapPoints和关键帧做bundle adjustment  全局优化
 *  这个全局BA优化在本程序中有两个地方使用：
 * a.单目初始化： CreateInitialMapMonocular    函数
 * b.闭环优化    ： RunGlobalBundleAdjustment 函数
 * @param pMap          全局地图
 * @param nIterations 优化迭代次数
 * @param pbStopFlag 设置是否强制暂停，需要终止迭代的时候强制终止 
 * @param nLoopKF     关键帧的个数
 * @param bRobust      是否使用核函数 鲁棒优化 (时间稍长)
 * 
 */  
    void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
	vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();// 地图的关键帧
	vector<MapPoint*> vpMP = pMap->GetAllMapPoints();// 地图的 地图点
	BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
    }

    // BA 最小化重投影误差     关键帧   地图点   优化迭代次数
    // 优化关键帧的位姿态 和  地图点坐标值
/**
 * @brief bundle adjustment Optimization
 * 3D-2D 最小化重投影误差 e = (u,v) - K * project(Tcw*Pw) \n
 * 
 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw  位姿
 *                  g2o::VertexSBAPointXYZ()，MapPoint 的 mWorldPos  地图点坐标
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge 二元边
 *         + Vertex 连接的顶点1：待优化当前帧的 位姿 Tcw
 *         + Vertex 连接的顶点2：待优化MapPoint的mWorldPos 地图点坐标
 *         + measurement 测量值(真实值)：MapPoint 地图点在当前帧中的二维位置(u,v)像素坐标
 *         + InfoMatrix信息矩阵(误差权重矩阵): invSigma2(与特征点所在的尺度有关)
 *         
 * @param   vpKFs          关键帧 
 * @param   vpMP           地图点 MapPoints
 * @param   nIterations 迭代次数（20次）
 * @param   pbStopFlag 是否强制暂停
 * @param   nLoopKF     关键帧的个数
 * @param   bRobust     是否使用核函数
 */   
    void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
				    int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
	vector<bool> vbNotIncludedMP;
	vbNotIncludedMP.resize(vpMP.size());//地图点数量
// 步骤1：初始化g2o优化器	
     //步骤1.1：设置求解器类型  帧位姿 pose 维度为 6 (优化变量维度), 地图点  landmark 维度为 3
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;// pose 维度为 6 (优化变量维度),  landmark 维度为 3
        // typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6 (优化变量维度),  landmark 维度为 3
	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();// 矩阵求解器 指针
	// linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
      //步骤1.2： 设置求解器
	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
      //步骤1.3：  设置函数优化方法
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);// LM 莱马算法
	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// 高斯牛顿
	// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//狗腿算法
       // 步骤1.4：设置稀疏优化求解器  
	g2o::SparseOptimizer optimizer;// 稀疏 优化模型
	optimizer.setAlgorithm(solver);// 设置求解器
        //   设置是否强制终止，需要终止迭代的时候强制终止 
	if(pbStopFlag)
	    optimizer.setForceStopFlag(pbStopFlag);// 优化停止标志

	long unsigned int maxKFid = 0;// 最大关键帧 ID
	
// 步骤2：向优化器添加顶点
	// Set KeyFrame vertices 
     // 步骤2.1：向优化器添加关键帧位姿顶点 添加位姿态顶点  设置 每一帧的 6自由度位姿 顶点
	for(size_t i=0; i<vpKFs.size(); i++)
	{
	    KeyFrame* pKF = vpKFs[i];//图中的每一个关键帧
	    if(pKF->isBad())//不好的帧 不优化  野帧
		continue;
	    // 顶点 vertex   优化变量
	    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();//camera pose   旋转矩阵 R   平移矩阵 t 的   李代数形式
	    vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose())); // 优化变量初始值  mat形式位姿 转成  SE3Quat 李代数形式
	    vSE3->setId(pKF->mnId);// 顶点 id 
	    vSE3->setFixed(pKF->mnId == 0);// 初始帧 位姿固定为 单位对角矩阵 世界坐标原点
	    optimizer.addVertex(vSE3);//添加顶点
	    if(pKF->mnId > maxKFid)
		maxKFid = pKF->mnId;// 最大关键帧 ID
	}

	const float thHuber2D = sqrt(5.99);	 //  g2o 优化为 两个值 像素点坐标               时鲁棒优化核函数 系数
	const float thHuber3D = sqrt(7.815);   //  g2o 优化为 3个值 像素点坐标 + 视差     时鲁棒优化核函数  系数

     // 步骤2.2：向优化器添加MapPoints顶点  添加3自由度 地图点 顶点
	// Set MapPoint vertices
	for(size_t i=0; i<vpMP.size(); i++)// 每一个地图点
	{
	    MapPoint* pMP = vpMP[i];// 地图点
	    if(pMP->isBad())//野点 跳过
		continue;
	    // g2o 3d 地图点类型
	    g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();// 3D 点  landmarks
	    vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));//优化变量初始值
	    const int id = pMP->mnId + maxKFid+1;//设置 3d 地图点 g2o图 中的 顶点 接着 位姿顶点id之后
	    vPoint->setId(id);// 顶点 id 
	    vPoint->setMarginalized(true);// 在优化过程中，这个节点应该被边缘化  g2o 中必须设置 marg
	    optimizer.addVertex(vPoint);// 添加顶点

	  //  const map<KeyFrame*,size_t> observations = pMP->GetObservations();
	  const auto observations = pMP->GetObservations();// 能够观测到该地图点的  观测关键帧 都应该和 这个地图顶点相连
	  // 地图点和地图点之间的 连线 是约束关系  是边 
	    int nEdges = 0;
	    
// 步骤3：向优化器添加投影边边  edge  地图点 和 各自观测帧 之间的 关系 
	    // map<KeyFrame*,size_t>::const_iterator mit 
	    for( map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
	    {

		KeyFrame* pKF = mit->first;//观测到该点的一个关键帧
		if(pKF->isBad() || pKF->mnId > maxKFid)//这个关键帧 是野帧 或者 不在优化的顶点范围内 跳过
		    continue;

		nEdges++;// 边 计数
		// 该地图点在 对应 观测帧 上  对应的 关键点 像素 坐标
		const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];// 该观测帧对应的 改点在 图像上的 像素点坐标

      // 【7】对于单目相机 右图匹配点坐标 小于0 的话是单目
		if( pKF->mvuRight[mit->second] < 0 )
		{
		    Eigen::Matrix<double,2,1> obs;//像素点坐标
		    obs << kpUn.pt.x, kpUn.pt.y;

		    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();// 边
	      // 设置 顶点 对应的地图点 和对于的 观测关键帧  相机位姿
		    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));// 对应的地图点
		    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));// 对应的关键帧
		    e->setMeasurement(obs);// 观测值是 帧上 对于的像素坐标
		    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];// 在图像金字塔上的层数
		    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);//信息矩阵  误差权重矩阵
		    // 观测值为 两个值 像素点坐标 所以 误差权重矩阵 为2*2
		    // 
    // 2D - 3D点对 优化方式  si * pi = K * T * Pi = K * [R t]* Pi  =  K * exp(f) * Pi  =  K * Pi'  
		    //   Pi'（地图点）为相机坐标系下的坐标  exp(f) * Pi  前三维 (Xi', Yi', Zi')     exp(f) 为 T的李代数形式
    /*  s*u       [fx 0 cx         X'
    *   s*v  =     0 fy cy  *     Y'
    *   s             0 0  1]         Z'
    *  利用第三行消去s(实际上就是 P'的深度)   u像素坐标
    *  u = fx * X'/Z' + cx    横坐标
    *  v = fy * Y'/Z'  + cy    纵坐标
    * 
    * p 观测值是 帧上 对于的像素坐标   u是地图点转换过来的
    *  * 误差 e  = p - u = p -K *P' 
    * e对∇f = e对u导数 * u对∇f 导数 = u对∇f 导数 = u对P'导数 * P'对∇f 导数         链式求导法则
    *
    *  * u对P'的偏导数 = - [ u对X'的偏导数 u对Y'的偏导数 u对Z'的偏导数;
    *                                   v对X'的偏导数 v对Y'的偏导数  v对Z'的偏导数]  = - [ fx/Z'   0        -fx * X'/Z' ^2 
    *                                                                                                                        0       fy/Z'    -fy* Y'/Z' ^2]
    *  *  P'对∇f的偏导数 = [ I  -P'叉乘矩阵] 3*6大小   平移在前  旋转在后
    *  = [ 1 0  0   0   Z'   -Y' 
    *       0 1  0  -Z'  0    X'
    *       0 0  1  Y'   -X   0]
    * 有向量 t = [ a1 a2 a3] 其
    * 叉乘矩阵 = [0  -a3  a2;
    *                     a3  0  -a1; 
    *                    -a2 a1  0 ]  
    * 
    * 两者相乘得到 
    * J = - [fx/Z'   0      -fx * X'/Z' ^2   -fx * X'*Y'/Z' ^2      fx + fx * X'^2/Z' ^2    -fx*Y'/Z'
    *           0     fy/Z'   -fy* Y'/Z' ^2    -fy -fy* Y'^2/Z' ^2   fy * X'*Y'/Z' ^2          fy*X'/Z'    ] 
    * 如果是 旋转在前 平移在后 调换前三列  后三列 
    * 
    * [2]  优化 P点坐标值
    * e 对P的偏导数   = e 对P'的偏导数 *  P'对P的偏导数 = e 对P'的偏导数 * R
    * P' = R * P + t
    * P'对P的偏导数  = R
    * 
    */		
		    if(bRobust)// 鲁棒优化核函数
		    {
			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);// 设置鲁棒优化核函数
			rk->setDelta(thHuber2D);
		    }
		    e->fx = pKF->fx;// 迭代 求雅克比所需参数
		    e->fy = pKF->fy;
		    e->cx = pKF->cx;
		    e->cy = pKF->cy;
		    optimizer.addEdge(e);
		}
      // 【9】对于双目相机和 深度相机      
		else
		{
		    Eigen::Matrix<double,3,1> obs;// 观测值 像素点坐标 以及 视差
		    const float kp_ur = pKF->mvuRight[mit->second];//深度得到的视差 立体匹配得到的视差
		    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

		    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();// 双目 边类型

		    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));// 对应的地图点
		    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));// 对应的关键帧 
		    e->setMeasurement(obs);// 观测值
		    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];// 误差权重
		    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;// 误差权重
		    // 观测值为 3个值 像素点坐标 和视差   所以 误差权重矩阵 为3*3
		    e->setInformation(Info);// 信息矩阵  误差权重矩阵

		    if(bRobust)
		    {
			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);// 鲁棒优化 核函数
			rk->setDelta(thHuber3D);// 系数
		    }

		    e->fx = pKF->fx;
		    e->fy = pKF->fy;
		    e->cx = pKF->cx;
		    e->cy = pKF->cy;
		    e->bf = pKF->mbf;// 视差优化 是 需要的 参数

		    optimizer.addEdge(e);
		}
	    }

	    if(nEdges==0)// 边的数量为0 没有地图点
	    {
		optimizer.removeVertex(vPoint);
		vbNotIncludedMP[i]=true;
	    }
	    else
	    {
		vbNotIncludedMP[i]=false;
	    }
	}
// 步骤4： 开始迭代优化
	// Optimize!
	optimizer.initializeOptimization();//初始化
	optimizer.optimize(nIterations);// 优化迭代

    // Recover optimized data
 // 步骤5：得到优化的结果 从优化结果 更新数据
	//Keyframes
     // 步骤5.1： 更新 帧 位姿
	for(size_t i=0; i<vpKFs.size(); i++)
	{
	    KeyFrame* pKF = vpKFs[i];
	    if(pKF->isBad())
		continue;
	    g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));// 位姿
	    g2o::SE3Quat SE3quat = vSE3->estimate();
	    if(nLoopKF==0)
	    {
		pKF->SetPose(Converter::toCvMat(SE3quat));// 更新 帧位姿
	    }
	    else
	    {
		pKF->mTcwGBA.create(4,4,CV_32F);
		Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
		pKF->mnBAGlobalForKF = nLoopKF;
	    }
	}
     // 步骤5.2： 更新地图点
	for(size_t i=0; i<vpMP.size(); i++)
	{
	    if(vbNotIncludedMP[i])
		continue;

	    MapPoint* pMP = vpMP[i];// 地图点

	    if(pMP->isBad())
		continue;
	    // 优化后的地图点
	    g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

	    if(nLoopKF==0)
	    {
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));// 设置 3d坐标
		pMP->UpdateNormalAndDepth();// 更新 距离相机中心距离 等信息
	    }
	    else
	    {
		pMP->mPosGBA.create(3,1,CV_32F);
		Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
		pMP->mnBAGlobalForKF = nLoopKF;
	    }
	}

    }


  
/**
 * @brief  仅仅优化 单个 普通帧 的位姿  地图点不优化  Pose Only Optimization
 * 当前帧跟踪参考关键帧 得到匹配点 和 设置上一帧位姿为初始姿态 进行优化 
 * 当前帧跟踪局部地图     得到匹配到的 地图点
 * 3D-2D 最小化重投影误差 e = (u,v) - project(Tcw*Pw) \n
 * 只优化Frame的Tcw，不优化MapPoints的坐标
 * 
 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZOnlyPose()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZOnlyPose()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 * @param   pFrame Frame
 * @return  inliers数量   返回 优化较好的 边数量  地图点和 帧上对应的二位像素坐标特征点
 */    
    int Optimizer::PoseOptimization(Frame *pFrame)
    {
      
// 步骤1：构造g2o优化器      
    //步骤1.1：设置求解器类型 帧位姿 pose 维度为 6 (优化变量维度), 地图点  landmark 维度为 3
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
	// linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
    //步骤1.2设置求解器
	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    //步骤1.3 设置函数优化方法
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// 高斯牛顿
	// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//狗腿算法	
    //步骤1.4 设置稀疏优化求解器  
	g2o::SparseOptimizer optimizer;// 稀疏 优化模型
	optimizer.setAlgorithm(solver);
	int nInitialCorrespondences=0;

	// Set Frame vertex
// 步骤2： 添加 位姿 顶点  设置 每一帧的 6自由度位姿 顶点
	g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
	vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));//初始值 相机位姿顶点
	vSE3->setId(0);
	vSE3->setFixed(false);
	optimizer.addVertex(vSE3);// 优化器 添加 位姿 顶点

	// 单目 边 类型
	const int N = pFrame->N;//  帧  的 地图点  个数
	vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;// 单目 边容器  保存边
	vector<size_t> vnIndexEdgeMono;
	vpEdgesMono.reserve(N);
	vnIndexEdgeMono.reserve(N);

	// 双目/深度边 类型
	vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;// 双目 / 深度 边容器 保存边
	vector<size_t> vnIndexEdgeStereo;
	vpEdgesStereo.reserve(N);
	vnIndexEdgeStereo.reserve(N);

	const float deltaMono = sqrt(5.991);
	const float deltaStereo = sqrt(7.815);

// 步骤3：添加一元边：相机投影模型
	{
	unique_lock<mutex> lock(MapPoint::mGlobalMutex);

	for(int i=0; i<N; i++)// 每个帧  的 地图点   
	{
	    MapPoint* pMP = pFrame->mvpMapPoints[i];//该帧上 的 每一个地图点
	    if(pMP)
	    {
		// Monocular observation
       // 单目情况, 也有可能在双目下, 当前帧的左兴趣点找不到匹配的右兴趣点 
	      // 添加仅 优化位姿 的 边 和 对应的  地图点(参数)
		if(pFrame->mvuRight[i]<0)// 右兴趣点坐标 
		{
		    nInitialCorrespondences++;//边的数量
		    pFrame->mvbOutlier[i] = false;
	   // 观测数据 像素点 坐标
		    Eigen::Matrix<double,2,1> obs;// 观测数据 像素坐标
		    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];//当前帧 关键点
		    obs << kpUn.pt.x, kpUn.pt.y;//
	   // 边 仅优化位姿   边 基础一元边  连接一个顶点
		    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
		    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));//   连接一个顶点
		    e->setMeasurement(obs);//测量值
		    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
		    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);//误差信息矩阵  权重

		    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		    e->setRobustKernel(rk);// 鲁棒优化核函数
		    rk->setDelta(deltaMono);

		    e->fx = pFrame->fx;// 参数
		    e->fy = pFrame->fy;
		    e->cx = pFrame->cx;
		    e->cy = pFrame->cy;
		    cv::Mat Xw = pMP->GetWorldPos();// 真实坐标点  作为参数提供 不作为优化变量
		    e->Xw[0] = Xw.at<float>(0);
		    e->Xw[1] = Xw.at<float>(1);
		    e->Xw[2] = Xw.at<float>(2);

		    optimizer.addEdge(e);

		    vpEdgesMono.push_back(e);// 容器保存 边
		    vnIndexEdgeMono.push_back(i);
		    //添加顶点  虚拟下标 为加入 地图点  地图点在 边e的参数中提供 不作为优化变量
		}
		
       //双目 添加仅 优化位姿 的 边 和 对应的  地图点(参数)           
		else  // Stereo observation
		{
		    nInitialCorrespondences++;//边的数量
		    pFrame->mvbOutlier[i] = false;

		    //SET EDGE
	    // 观测值  像素点坐标 +  视差
		    Eigen::Matrix<double,3,1> obs;
		    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
		    const float &kp_ur = pFrame->mvuRight[i];
		    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
	    //  边 基础一元边  连接一个顶点
		    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
		    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));// 1个顶点
		    e->setMeasurement(obs);
		    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
		    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
		    // 观测值为 3个值 像素点坐标 和视差   所以 误差权重矩阵 为3*3
		    e->setInformation(Info);

		    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		    e->setRobustKernel(rk);// 鲁棒优化核函数
		    rk->setDelta(deltaStereo);

		    e->fx = pFrame->fx;// 参数
		    e->fy = pFrame->fy;
		    e->cx = pFrame->cx;
		    e->cy = pFrame->cy;
		    e->bf = pFrame->mbf;// 视差所需参数
		    cv::Mat Xw = pMP->GetWorldPos();//地图点 作为参数  不作为优化变量
		    e->Xw[0] = Xw.at<float>(0);
		    e->Xw[1] = Xw.at<float>(1);
		    e->Xw[2] = Xw.at<float>(2);

		    optimizer.addEdge(e);// 添加边

		    vpEdgesStereo.push_back(e);
		    vnIndexEdgeStereo.push_back(i);
		}
	    }

	}
	}


	if(nInitialCorrespondences<3)//该顶点 边的数量 小于3个  不优化
	    return 0;

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
// 步骤4：开始优化，总共优化四次，每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
        // 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然
        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
	const float chi2Mono[4]={5.991,5.991,5.991,5.991};
	const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
	const int its[4]={10,10,10,10};    // 优化10次
	int nBad=0;
	for(size_t it=0; it<4; it++)
	{
	    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));//初始值
	    optimizer.initializeOptimization(0);// 对level为0的边进行优化
	    optimizer.optimize(its[it]);

	    nBad=0;
    // wyw  2018 添加	
	    // 单目 更新外点标志
	//    if(pFrame->mvuRight[1]<0)// 匹配点坐标 小于0 的话是单目
	 //   {	
		    for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)// 单目 每一条边
		    {
			g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];//

			const size_t idx = vnIndexEdgeMono[i];

			if(pFrame->mvbOutlier[idx])//外点
			{
			    e->computeError();// NOTE g2o只会计算active edge的误差 
			}
			const float chi2 = e->chi2();
			if(chi2 > chi2Mono[it])
			{                
			    pFrame->mvbOutlier[idx] = true;// 确实是外点 不好的点
			    e->setLevel(1);// 设置为outlier
			    nBad++;
			}
			else
			{
			    pFrame->mvbOutlier[idx] = false;// 原来是外点  优化过后 误差变小了  变成内点了
			    e->setLevel(0);// 设置为inlier
			}

			if(it==2)
			    e->setRobustKernel(0);// 除了前两次优化需要RobustKernel以外, 其余的优化都不需要
		    }
	  //}
	  //  else{
		      for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
		      {
			  g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

			  const size_t idx = vnIndexEdgeStereo[i];

			  if(pFrame->mvbOutlier[idx])
			  {
			      e->computeError();
			  }

			  const float chi2 = e->chi2();

			  if(chi2>chi2Stereo[it])
			  {
			      pFrame->mvbOutlier[idx]=true;
			      e->setLevel(1);
			      nBad++;
			  }
			  else
			  {                
			      e->setLevel(0);
			      pFrame->mvbOutlier[idx]=false;
			  }

			  if(it==2)// 除了前两次优化需要RobustKernel以外, 其余的优化都不需要
			      e->setRobustKernel(0);
		      }
	  //  }
	    if(optimizer.edges().size()<10)
		break;
      }    

	// Recover optimized pose and return number of inliers
// 步骤5：优化过后 更新 帧 位姿
	g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
	g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
	cv::Mat pose = Converter::toCvMat(SE3quat_recov);
	pFrame->SetPose(pose);

	return nInitialCorrespondences-nBad;
    }

    
/**
 * @brief Local Bundle Adjustment 局部地图优化 删除一些优化误差大的 帧/地图点 观测对
 *
 * 1. Vertex:
 *     - g2o::VertexSE3Expmap()， 局部关键帧集合 LocalKeyFrames ，即当前关键帧的位姿、与当前关键帧一级相连的关键帧的位姿
 *     - g2o::VertexSE3Expmap()， 局部固定帧 FixedCameras ，即能观测到 局部地图点 LocalMapPoints 的关键帧
 *                                                  但不属于 局部关键帧 LocalKeyFrames 的关键帧，在优化中这些关键帧的位姿不变
 *     - g2o::VertexSBAPointXYZ()， 局部地图点集 LocalMapPoints，即 LocalKeyFrames 能观测到的所有MapPoints的位置
 * 
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge   
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *         
 * @param pKF              KeyFrame
 * @param pbStopFlag 是否停止优化的标志
 * @param pMap          在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
 */
    void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
    {    
	// 本地关键帧  Local KeyFrames: First Breath Search from Current Keyframe
	list<KeyFrame*> lLocalKeyFrames;//  局部关键帧集合 关键帧 的一级相邻帧 集合
	
// 步骤1：将当前关键帧加入 局部关键帧集合 lLocalKeyFrames
	lLocalKeyFrames.push_back(pKF);
	pKF->mnBALocalForKF = pKF->mnId;
// 步骤2：找到关键帧连接的关键帧（一级相连），加入 lLocalKeyFrames 中	
       // 寻找关键帧 的 一级相邻帧
	const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();//关键帧 的 一级相邻帧
	for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
	{
	    KeyFrame* pKFi = vNeighKFs[i];//关键帧 的 一级相邻帧
	    pKFi->mnBALocalForKF = pKF->mnId;
	    if(!pKFi->isBad())
		lLocalKeyFrames.push_back(pKFi);// 当前帧 及其 一级相邻帧
	}
	
// 步骤3：遍历 lLocalKeyFrames 中关键帧，将它们观测的MapPoints加入到 局部地图点集 lLocalMapPoints
	// 局部地图点集  Local MapPoints seen in Local KeyFrames
	list<MapPoint*> lLocalMapPoints;// 局部地图点集
	// 遍历每一个 局部关键帧
        // list<KeyFrame*>::iterator  
	for(auto lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
	{
	    vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();// 每个局部关键帧的 地图点
	    // 遍历 每一个 局部关键帧 的 每一个 地图点
	    // vector<MapPoint*>::iterator 
	    for(auto vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
	    {
		MapPoint* pMP = *vit;// 每一个 局部关键帧 的 每一个 地图点
		if(pMP)
		    if(!pMP->isBad())
			if(pMP->mnBALocalForKF != pKF->mnId)// 避免重复添加
			{
			    lLocalMapPoints.push_back(pMP);// 添加到 局部地图点集 
			    pMP->mnBALocalForKF=pKF->mnId;// 标记已经加入点集
			}
	    }
	}

	//固定关键帧  Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
// 步骤4：能观测到局部地图点集地图点的关键帧，但不属于局部关键帧的关键帧，这些关键帧在局部BA优化时不优化	
	list<KeyFrame*> lFixedCameras;
	// 遍历 局部地图点集 中的 每一个地图点 查看其观测帧
	// list<MapPoint*>::iterator
	for(auto lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
	{
	    map<KeyFrame*,size_t> observations = (*lit)->GetObservations();//局部地图点 的 观测帧
	    // 遍历每一个 局部地图点 的 观测帧  查看其是否在 局部关键帧中
	    // map<KeyFrame*,size_t>::iterator
	    for(auto mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
	    {
		KeyFrame* pKFi = mit->first;// 每一个 局部地图点 的 观测帧
                // 局部关键帧 不包含   局部地图点的观测帧 并且 局部关键帧 未被 添加     避免重复添加
		if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
		{                
		    pKFi->mnBAFixedForKF = pKF->mnId;// 标记  局部固定帧
		    if(!pKFi->isBad())
			lFixedCameras.push_back(pKFi);// 添加 局部固定帧
		}
	    }
	}
// 步骤5：构造g2o优化器
	// 启动优化器  Setup optimizer
     // 步骤5.1：求解器类型   帧位姿 pose 维度为 6 (优化变量维度), 地图点  landmark 维度为 3
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
     // 步骤5.2：迭代优化算法  
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// 高斯牛顿
	// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//狗腿算法	
     // 步骤5.3：设置优化器   
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);
     //   设置是否强制终止，需要终止迭代的时候强制终止 
	if(pbStopFlag)
	    optimizer.setForceStopFlag(pbStopFlag);

	unsigned long maxKFid = 0;

// 步骤6：添加顶点 局部关键帧 位姿 顶点 Set Local KeyFrame vertices
	//list<KeyFrame*>::iterator lit
	for(auto lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
	{
	    KeyFrame* pKFi = *lit;//  局部关键帧
	    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();//顶点类型
	    vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));// 初始值
	    vSE3->setId(pKFi->mnId);//id
	    vSE3->setFixed(pKFi->mnId == 0);// 第一帧关键帧 固定
	    optimizer.addVertex(vSE3);// 添加顶点
	    if(pKFi->mnId > maxKFid)
		maxKFid = pKFi->mnId;// 局部关键帧 最大的
	}

// 步骤7：添加顶点：设置固定关键帧顶点  Set Fixed KeyFrame vertices
        // list<KeyFrame*>::iterator lit
	for(auto lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
	{
	    KeyFrame* pKFi = *lit;// 局部固定关键帧
	    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();//顶点类型
	    vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
	    vSE3->setId(pKFi->mnId);
	    vSE3->setFixed(true);//位姿 固定 不变
	    optimizer.addVertex(vSE3);
	    if(pKFi->mnId > maxKFid)
		maxKFid = pKFi->mnId;
	}

// 步骤8：设置 地图点 顶点 Set MapPoint vertices 帧和每一个地图点都可能相连形成边
	const int nExpectedSize = ( lLocalKeyFrames.size() + lFixedCameras.size() ) * lLocalMapPoints.size();

	vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;// 单目地图点 边类型  
	vpEdgesMono.reserve(nExpectedSize);
	vector<KeyFrame*> vpEdgeKFMono;// 单目关键帧
	vpEdgeKFMono.reserve(nExpectedSize);

	vector<MapPoint*> vpMapPointEdgeMono;// 双目地图点
	vpMapPointEdgeMono.reserve(nExpectedSize);// 
	vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;// 双目 地图点 边
	vpEdgesStereo.reserve(nExpectedSize);

	vector<KeyFrame*> vpEdgeKFStereo;// 双目 关键帧   
	vpEdgeKFStereo.reserve(nExpectedSize);
	vector<MapPoint*> vpMapPointEdgeStereo;// 双目 地图点 
	vpMapPointEdgeStereo.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);
	// list<MapPoint*>::iterator lit
      // 遍历 每一个 局部地图点
	for(auto lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
	{
	    // 添加顶点：MapPoint
	    MapPoint* pMP = *lit;//每一个 局部地图点
	    g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();// 地图点 顶点类型
	    vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));// 初始值
	    int id = pMP->mnId + maxKFid + 1;//在顶点id 后
	    vPoint->setId(id);
	    vPoint->setMarginalized(true);
	    optimizer.addVertex(vPoint);// 添加顶点

	    const map<KeyFrame*,size_t> observations = pMP->GetObservations();// 地图点对应的 观测 帧

// 步骤9：对每一对关联的MapPoint和KeyFrame构建边   
	    // map<KeyFrame*,size_t>::const_iterator mit
	    for(auto  mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
	    {
		KeyFrame* pKFi = mit->first;// 每一个顶点的 观测关键帧帧 

		if(!pKFi->isBad())
		{                
		    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];// 观测帧 对应 该地图点 的 像素坐标

	 // 步骤9.1：单目下 添加边 Monocular observation  单目观测下 观测值 像素坐标
		    if(pKFi->mvuRight[mit->second] < 0)
		    {
		     // 观测值  两维 像素坐标
			Eigen::Matrix<double,2,1> obs;
			obs << kpUn.pt.x, kpUn.pt.y;
		     // 二元边
			g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();// 二元边  两个顶点
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));// 地图点
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));// 帧 位姿
			e->setMeasurement(obs);//观测值
			const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
			e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);// 误差信息矩阵
		     // 核函数
			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(thHuberMono);
		     // 参数信息 用于 雅克比矩阵    迭代优化
			e->fx = pKFi->fx;
			e->fy = pKFi->fy;
			e->cx = pKFi->cx;
			e->cy = pKFi->cy;

			optimizer.addEdge(e);//添加边
			vpEdgesMono.push_back(e);//单目边
			vpEdgeKFMono.push_back(pKFi);//关键帧
			vpMapPointEdgeMono.push_back(pMP);// 地图点
		    }		    
	   //   步骤9.2：双目下 添加边  // Stereo observation 观测值 像素坐标和 视差
		    else 
		    {
		    // 观测值 像素坐标和 视差
			Eigen::Matrix<double,3,1> obs;
			const float kp_ur = pKFi->mvuRight[mit->second];
			obs << kpUn.pt.x, kpUn.pt.y, kp_ur;// 像素坐标和 视差
		      // 二元边
			g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
			e->setMeasurement(obs);
			const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
			Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
			e->setInformation(Info);
		      //  核函数
			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(thHuberStereo);
		    // 参数信息 用于 雅克比矩阵    迭代优化
			e->fx = pKFi->fx;
			e->fy = pKFi->fy;
			e->cx = pKFi->cx;
			e->cy = pKFi->cy;
			e->bf = pKFi->mbf;
		    // 添加边
			optimizer.addEdge(e);// 添加边
			vpEdgesStereo.push_back(e);// 保存 边信息
			vpEdgeKFStereo.push_back(pKFi);// 关键帧
			vpMapPointEdgeStereo.push_back(pMP);// 地图点 
		    }
		}
	    }
	}

	if(pbStopFlag)
	    if(*pbStopFlag)
		return;
// 步骤10：开始优化 
	optimizer.initializeOptimization();
	optimizer.optimize(5);// 优化5次

	bool bDoMore= true;

	if(pbStopFlag)
	    if(*pbStopFlag)
		bDoMore = false;

	if(bDoMore)//需要排除误差较大的点 再优化 10次
	{
// 步骤11：检测outlier（误差过大  外点），并设置下次不优化
	    // Check inlier observations
	    // 更新内点 标志
	    //  if(pKFi->mvuRight[1]<0)
	    //  {  
	  
	 // 步骤11.1：遍历每一个单目 优化边
	      for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
	      {
		  g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
		  MapPoint* pMP = vpMapPointEdgeMono[i];// 边对应的地图点

		  if(pMP->isBad())
		      continue;
		  // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
		  if(e->chi2() > 5.991 || !e->isDepthPositive())
		  {
		      e->setLevel(1);// 不优化
		  }
		  e->setRobustKernel(0);// 不使用核函数
	      }
	      
         // 步骤11.2 : 遍历每一个双目 优化边
	      for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
	      {
		  g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
		  MapPoint* pMP = vpMapPointEdgeStereo[i];

		  if(pMP->isBad())
		      continue;
                 // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
		  if(e->chi2() > 7.815 || !e->isDepthPositive())
		  {
		      e->setLevel(1);// 不优化
		  }

		  e->setRobustKernel(0);// 不使用核函数
	      }

	// Optimize again without the outliers
// 步骤12：排除误差较大的outlier后再次优化 10次
	optimizer.initializeOptimization(0);
	optimizer.optimize(10);

      }
      
 // 步骤13：在优化后重新计算误差，剔除连接误差比较大的关键帧和MapPoint
	vector<pair<KeyFrame*,MapPoint*> > vToErase;// 连接误差较大 需要 剔除的 关键帧和MapPoint
	vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());//单目边 双目边
	    // Check inlier observations 
	 // 每一个单目边  误差 两维 
	    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
	    {
		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];//单目优化边
		MapPoint* pMP = vpMapPointEdgeMono[i];// 边对应的 地图点

		if(pMP->isBad())
		    continue;
               // 基于卡方检验计算出的阈值（误差过大，删除 假设测量有一个像素的偏差）
		if(e->chi2()>5.991 || !e->isDepthPositive())
		{
		    KeyFrame* pKFi = vpEdgeKFMono[i];//边对应的 帧
       // 步骤13.1：标记需要删除的边	    
		    vToErase.push_back(make_pair(pKFi,pMP));//删除 这个边 
		}
	    }
          // 每一个双目边  误差 三维 
	    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
	    {
		g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];//双目优化边
		MapPoint* pMP = vpMapPointEdgeStereo[i];// 边对应的 地图点

		if(pMP->isBad())
		    continue;

		if(e->chi2()>7.815 || !e->isDepthPositive())
		{
		    KeyFrame* pKFi = vpEdgeKFStereo[i];//边对应的 帧
       // 步骤13.1：标记需要删除的边	    
		    vToErase.push_back(make_pair(pKFi,pMP));//删除 这个边 
		}
	    }
	    
       //步骤13.2 删除 误差较大的 边
         // 连接偏差比较大，在关键帧中剔除对该MapPoint的观测
         // 连接偏差比较大，在MapPoint中剔除对该关键帧的观测
	 // Get Map Mutex
	unique_lock<mutex> lock(pMap->mMutexMapUpdate);
	if(!vToErase.empty())
	{
	    for(size_t i=0;i<vToErase.size();i++)
	    {
		KeyFrame* pKFi = vToErase[i].first;// 边对应的 帧
		MapPoint* pMPi = vToErase[i].second;// 边对应的 地图点
		pKFi->EraseMapPointMatch(pMPi);//帧 删除 地图点    观测
		pMPi->EraseObservation(pKFi);// 地图点 删除 观测帧 观测
	    }
	}

	// Recover optimized data
// 步骤14：优化后更新关键帧位姿以及MapPoints的位置、平均观测方向等属性
     // 步骤14.1：优化后更新 关键帧 Keyframes
	// list<KeyFrame*>::iterator
	for(auto  lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
	{
	    KeyFrame* pKF = *lit;//关键帧
	    g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
	    g2o::SE3Quat SE3quat = vSE3->estimate();
	    pKF->SetPose(Converter::toCvMat(SE3quat));
	}

     //步骤14.2：优化后 更新地图点 Points
        // list<MapPoint*>::iterator
	for(auto lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
	{
	    MapPoint* pMP = *lit;//地图点
	    g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid+1));
	    pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));//更新位置
	    pMP->UpdateNormalAndDepth();// 更新 平均观测方向 深度
	}
    }

    
/**
 * @brief 闭环检测后，关键帧连接图 EssentialGraph 优化
 *
 * 1. Vertex:
 *     - g2o::VertexSim3Expmap ， Essential graph 中关键帧的位姿
 * 2. Edge:
 *     - g2o::EdgeSim3()，BaseBinaryEdge      基础二元边
 *         + Vertex： 关键帧的 位姿 Tcw， 地图点MapPoint的 位置 Pw
 *         + measurement：经过CorrectLoop函数步骤2，Sim3传播校正后的位姿
 *         + InfoMatrix: 单位矩阵     
 *
 * @param pMap                 全局地图
 * @param pLoopKF            闭环匹配上的关键帧
 * @param pCurKF              当前关键帧
 * @param NonCorrectedSim3    未经过Sim3传播调整过的关键帧 位姿 对
 * @param CorrectedSim3           经过Sim3传播调整过的关键帧 位姿 对
 * @param LoopConnections       因闭环时 MapPoints 调整而新生成的边
 * @param bFixScale                     固定尺度大小   
 */
    void Optimizer::OptimizeEssentialGraph(
					  Map* pMap, 
					  KeyFrame* pLoopKF, 
					  KeyFrame* pCurKF,
					  const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
					  const LoopClosing::KeyFrameAndPose &CorrectedSim3,
					  const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
    {
      
// 步骤1：构造优化器      
	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
     // 步骤1.1：求解器类型   帧sim3位姿 spose 维度为 7  [sR t], 地图点  landmark 维度为 3
	 //  指定线性方程求解器使用Eigen的块求解器
	g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
	      new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
     // 步骤1.2：构造线性求解器
	g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
     // 步骤1.3：迭代优化算法  
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);// 使用LM算法进行非线性迭代
	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// 高斯牛顿
	// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//狗腿算法	
     // 步骤1.4：设置优化器  	
	solver->setUserLambdaInit(1e-16);
	optimizer.setAlgorithm(solver);

	const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();//全局地图 的 所有 关键帧 
	const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();//全局地图 的 所有 地图点

	const unsigned int nMaxKFid = pMap->GetMaxKFid();//最大关键帧 id
        // 仅经过Sim3传播调整，未经过优化的keyframe的pose
	vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);// 存储 优化前 帧 的位姿
	// 经过Sim3传播调整，经过优化的keyframe的pose
	vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);//  存储 优化后 帧 的位姿
	//  
	vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);//保存g2o 顶点

	const int minFeat = 100;
	
// 步骤2：将地图中所有keyframe的pose作为顶点添加到优化器
         // 尽可能使用经过Sim3调整的位姿
        // 关键帧顶点 Set KeyFrame vertices
	for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
	{
	    KeyFrame* pKF = vpKFs[i];//关键帧
	    if(pKF->isBad())
		continue;
	    g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();// 顶点类型 sim3 相似变换
	    const int nIDi = pKF->mnId;//关键帧 id
	    LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);//查看该关键帧是否在 闭环优化帧内 
	    
       //步骤2.1： 如果该关键帧 在闭环时通过Sim3传播调整过，用校正后的位姿
	    if(it != CorrectedSim3.end())
	    {
		vScw[nIDi] = it->second;// Sim3传播调整过的位姿 
		VSim3->setEstimate(it->second);//设置 顶点初始 估计值
	    }
       //步骤2.2： 如果该关键帧在闭环时没有通过Sim3传播调整过，用自身的位姿
	    else
	    {
		Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
		Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
		g2o::Sim3 Siw(Rcw,tcw,1.0);//设置体积估计尺度为1
		vScw[nIDi] = Siw;//存储帧 对应的 sim3 位姿
		VSim3->setEstimate(Siw);//设置 顶点初始 估计值
	    }
       //步骤2.3： 闭环匹配上的帧不进行位姿优化
	    if(pKF == pLoopKF)//  闭环匹配上的关键帧
		VSim3->setFixed(true);// 固定不优化

	    VSim3->setId(nIDi);//顶点id
	    VSim3->setMarginalized(false);//
	    VSim3->_fix_scale = bFixScale; // 固定尺度大小  

	    optimizer.addVertex(VSim3);// 添加 顶点

	    vpVertices[nIDi]=VSim3;//保存顶点
	}


	set<pair<long unsigned int,long unsigned int> > sInsertedEdges;
	const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();//信息矩阵

	// Set Loop edges
// 步骤3：添加闭环新边( 帧 连接 帧 )：LoopConnections是闭环时因为MapPoints调整而出现的新关键帧连接关系（不是当前帧与闭环匹配帧之间的连接关系）	
        //  遍历  因闭环时 MapPoints 调整而新生成的边
	for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
	{
	    KeyFrame* pKF = mit->first;//关键帧 
	    const long unsigned int nIDi = pKF->mnId;//id
	    const set<KeyFrame*> &spConnections = mit->second;// 与关键帧 相连的 关键帧
	    const g2o::Sim3 Siw = vScw[nIDi];//顶点帧 位姿
	    const g2o::Sim3 Swi = Siw.inverse();// 逆

	    for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit != send; sit++)
	    {
		const long unsigned int nIDj = (*sit)->mnId;// 相连关键帧 id
		if((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat)
		    continue;

		const g2o::Sim3 Sjw = vScw[nIDj];
		// 得到两个pose间的Sim3变换
		const g2o::Sim3 Sji = Sjw * Swi;//

		g2o::EdgeSim3* e = new g2o::EdgeSim3();// 边类型
		e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
		e->setMeasurement(Sji);//测量值

		e->information() = matLambda;//信息矩阵

		optimizer.addEdge(e);//添加边

		sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
	    }
	}

// 步骤4：添加跟踪时形成的边、闭环匹配成功形成的边  Set normal edges
	for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
	{
	    KeyFrame* pKF = vpKFs[i];// 关键帧

	    const int nIDi = pKF->mnId;

	    g2o::Sim3 Swi;
	    
            // 尽可能得到未经过Sim3传播调整的位姿
	    LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);
	    if(iti != NonCorrectedSim3.end())
		Swi = (iti->second).inverse();// 未经过Sim3传播调整的位姿
	    else
		Swi = vScw[nIDi].inverse();//  经过Sim3传播调整的位姿
		
     // 步骤4.1： 父子边  只添加扩展树的边（有父关键帧） 父关键帧<----->关键帧
	    KeyFrame* pParentKF = pKF->GetParent();
	    // Spanning tree edge
	    if(pParentKF)
	    {
		int nIDj = pParentKF->mnId;

		g2o::Sim3 Sjw;

		LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);
              // 尽可能得到未经过Sim3传播调整的位姿
		if(itj!=NonCorrectedSim3.end())
		    Sjw = itj->second;// 未经过Sim3传播调整的位姿
		else
		    Sjw = vScw[nIDj];//  经过Sim3传播调整的位姿
               // 父子 位姿 变换 
		g2o::Sim3 Sji = Sjw * Swi;

		g2o::EdgeSim3* e = new g2o::EdgeSim3();// 普通边
		e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
		e->setMeasurement(Sji);
		e->information() = matLambda;// 信息矩阵 误差权重矩阵
		optimizer.addEdge(e);//添加边
	    }
	    
     // 步骤4.2：关键帧<---->闭环帧 添加在CorrectLoop函数中AddLoopEdge函数添加的闭环连接边（当前帧与闭环匹配帧之间的连接关系）
            // 使用经过Sim3调整前关键帧之间的相对关系作为边
	    // Loop edges
	    const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
	    for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
	    {
		KeyFrame* pLKF = *sit;
		if(pLKF->mnId<pKF->mnId)
		{
		    g2o::Sim3 Slw;

		    LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);
                   // 尽可能得到未经过Sim3传播调整的位姿
		    if(itl!=NonCorrectedSim3.end())
			Slw = itl->second;
		    else
			Slw = vScw[pLKF->mnId];

		    g2o::Sim3 Sli = Slw * Swi;
		    g2o::EdgeSim3* el = new g2o::EdgeSim3();
		    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
		    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
		    // 根据两个Pose顶点的位姿算出相对位姿作为边 初始值 
		    el->setMeasurement(Sli);
		    el->information() = matLambda;
		    optimizer.addEdge(el);
		}
	    }
	    
       // 步骤4.3：关键帧<----->相邻帧  最有很好共视关系的关键帧也作为边进行优化
            // 使用经过Sim3调整前关键帧之间的相对关系作为边
	    // Covisibility graph edges
	    const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);// 100个相邻帧
	    for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
	    {
		KeyFrame* pKFn = *vit;// 关键帧 相邻帧
		// 非 父子帧边 无孩子  无闭环边
		if(pKFn && pKFn !=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
		{
		    if(!pKFn->isBad() && pKFn->mnId < pKF->mnId)
		    {
			if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
			    continue;

			g2o::Sim3 Snw;

			LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);
			
                        // 尽可能得到未经过Sim3传播调整的位姿
			if(itn!=NonCorrectedSim3.end())
			    Snw = itn->second;
			else
			    Snw = vScw[pKFn->mnId];

			g2o::Sim3 Sni = Snw * Swi;

			g2o::EdgeSim3* en = new g2o::EdgeSim3();
			en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
			en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
			en->setMeasurement(Sni);
			en->information() = matLambda;// 信息矩阵
			optimizer.addEdge(en);
		    }
		}
	    }
	}
// 步骤5：开始g2o优化
	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(20);//优化20次

	unique_lock<mutex> lock(pMap->mMutexMapUpdate);

	// SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
// 步骤6：设定帧关键帧优化后的位姿
	for(size_t i=0;i<vpKFs.size();i++)
	{
	    KeyFrame* pKFi = vpKFs[i];//关键帧

	    const int nIDi = pKFi->mnId;

	    g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
	    g2o::Sim3 CorrectedSiw =  VSim3->estimate();// 优化后的 关键帧的 sim3位姿
	    vCorrectedSwc[nIDi]=CorrectedSiw.inverse(); // 存入 优化后的位姿
	    Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
	    Eigen::Vector3d eigt = CorrectedSiw.translation();
	    double s = CorrectedSiw.scale();// 尺度

	    eigt *=(1./s); //[R t/s;0 1]
	    cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);
	    pKFi->SetPose(Tiw);// 欧式变换位姿
	}
// 步骤7：步骤5和步骤6优化得到 关键帧的位姿 后，MapPoints根据参考帧 优化前后的相对关系调整自己的位置
	// Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
	for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
	{
	    MapPoint* pMP = vpMPs[i];//地图点

	    if(pMP->isBad())
		continue;

	    int nIDr;
	    // 该MapPoint经过Sim3调整过，(LoopClosing.cpp，CorrectLoop函数，步骤2.2_
	    if(pMP->mnCorrectedByKF == pCurKF->mnId)
	    {
		nIDr = pMP->mnCorrectedReference;
	    }
	    else
	    {
	      // 通过情况下MapPoint的参考关键帧就是创建该MapPoint的那个关键帧
		KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
		nIDr = pRefKF->mnId;// 地图点的 参考帧 id
	    }

             // 得到MapPoint参考关键帧步骤5优化前的位姿
	    g2o::Sim3 Srw = vScw[nIDr];// 地图点参考帧 优化前的 位姿
	     // 得到MapPoint参考关键帧优化后的位姿
	    g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];// 地图点参考帧 优化后的 位姿

	    cv::Mat P3Dw = pMP->GetWorldPos();// 地图点 原坐标
	    Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);//转成 eigen
	    // 先按优化前位姿从世界坐标系 转到 当前帧下 再按 优化后的位姿 转到世界坐标系下
	    Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

	    cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);//转成opencv mat
	    pMP->SetWorldPos(cvCorrectedP3Dw);//设置更新后的 坐标
	    pMP->UpdateNormalAndDepth();// 更新 地图点  平均观测方向 深度
	}
    }
    
/**
 * @brief 形成闭环时进行Sim3优化   优化 两个关键帧 之间的 Sim3变换
 *
 * 1. 顶点 Vertex:
 *     - g2o::VertexSim3Expmap()，两个关键帧的位姿
 *     - g2o::VertexSBAPointXYZ()，两个关键帧共有的MapPoints
 * 
 * 2. 边 Edge:
 *     - g2o::EdgeSim3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Sim3，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 * 
 *     - g2o::EdgeInverseSim3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Sim3，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *         
 * @param pKF1        KeyFrame
 * @param pKF2        KeyFrame
 * @param vpMatches1  两个关键帧的匹配关系
 * @param g2oS12          两个关键帧间的Sim3变换
 * @param th2                 核函数阈值
 * @param bFixScale       是否优化尺度，弹目进行尺度优化，双目不进行尺度优化
 */
    int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
    {
// 步骤1：初始化g2o优化器
     // 先构造求解器
	g2o::SparseOptimizer optimizer;
     // 构造线性方程求解器，Hx = -b的求解器
	// typedef BlockSolver< BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> > BlockSolverX
	g2o::BlockSolverX::LinearSolverType * linearSolver;
     // 使用dense的求解器，（常见非dense求解器有cholmod线性求解器和shur补线性求解器）
	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
	g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
     // 使用 L-M 迭代   迭代优化算法  
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);// 使用LM算法进行非线性迭代
	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// 高斯牛顿
	// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//狗腿算法	
     //  设置优化器  	
	optimizer.setAlgorithm(solver);

	// 相机内参数 Calibration
	const cv::Mat &K1 = pKF1->mK;
	const cv::Mat &K2 = pKF2->mK;

	// 相机位姿 Camera poses
	const cv::Mat R1w = pKF1->GetRotation();
	const cv::Mat t1w = pKF1->GetTranslation();
	const cv::Mat R2w = pKF2->GetRotation();
	const cv::Mat t2w = pKF2->GetTranslation();
	
// 步骤2：添加相似Sim3顶点
	// Set Sim3 vertex
	g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();  // sim3 顶点类型
	vSim3->_fix_scale=bFixScale;
	vSim3->setEstimate(g2oS12);// 初始估计值 两帧 之间的 相似变换
	vSim3->setId(0);//id
	vSim3->setFixed(false);// 优化Sim3顶点
	vSim3->_principle_point1[0] = K1.at<float>(0,2);// 光心横坐标 cx
	vSim3->_principle_point1[1] = K1.at<float>(1,2);// 光心纵坐标 cy
	vSim3->_focal_length1[0] = K1.at<float>(0,0);// 焦距 fx
	vSim3->_focal_length1[1] = K1.at<float>(1,1);// 焦距 fy
	vSim3->_principle_point2[0] = K2.at<float>(0,2);
	vSim3->_principle_point2[1] = K2.at<float>(1,2);
	vSim3->_focal_length2[0] = K2.at<float>(0,0);
	vSim3->_focal_length2[1] = K2.at<float>(1,1);
	optimizer.addVertex(vSim3);// 添加顶点
	
// 步骤3 ：添加 地图点 顶点
       // Set MapPoint vertices
	const int N = vpMatches1.size();// 帧2 的匹配地图点
	const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
	vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;             //pKF2对应的MapPoints到pKF1的投影
	vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;//pKF1对应的MapPoints到pKF2的投影
	vector<size_t> vnIndexEdge;

	vnIndexEdge.reserve(2*N);
	vpEdges12.reserve(2*N);
	vpEdges21.reserve(2*N);

	const float deltaHuber = sqrt(th2);

	int nCorrespondences = 0;

	for(int i=0; i<N; i++)
	{
	    if(!vpMatches1[i])
		continue;
	    
            // pMP1和pMP2是匹配的MapPoints
	    MapPoint* pMP1 = vpMapPoints1[i];// 帧1 地图点
	    MapPoint* pMP2 = vpMatches1[i];    // 帧2 地图点

	    const int id1 = 2*i+1;
	    const int id2 = 2*(i+1);

	    const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

	    if(pMP1 && pMP2)
	    {
		if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
		{
      // 步骤3.1 添加PointXYZ顶点		  
		    g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
		    cv::Mat P3D1w = pMP1->GetWorldPos();
		    cv::Mat P3D1c = R1w*P3D1w + t1w;
		    vPoint1->setEstimate(Converter::toVector3d(P3D1c));// 帧1 下的点坐标
		    vPoint1->setId(id1);// 帧1  地图点
		    vPoint1->setFixed(true);
		    optimizer.addVertex(vPoint1);

		    g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
		    cv::Mat P3D2w = pMP2->GetWorldPos();
		    cv::Mat P3D2c = R2w*P3D2w + t2w;
		    vPoint2->setEstimate(Converter::toVector3d(P3D2c));// 帧2下的点坐标
		    vPoint2->setId(id2);// 帧2 地图点
		    vPoint2->setFixed(true);
		    optimizer.addVertex(vPoint2);
		}
		else
		    continue;
	    }
	    else
		continue;

	    nCorrespondences++;
	    
// 步骤4 ： 添加两个顶点（3D点）到相机投影的边
       // 步骤4.1 ：添加 帧2 地图点 映射到 帧1特征点 的 边
	    // Set edge x1 = S12*X2
	    Eigen::Matrix<double,2,1> obs1;
	    const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
	    obs1 << kpUn1.pt.x, kpUn1.pt.y;// 帧1 特征点 的 实际值
	    
	    g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();// 边类型
	    e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));// 帧2 地图点
	    e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));// 帧2 到 帧1 变换顶点
	    e12->setMeasurement(obs1);// 帧1 特征点 的 实际值
	    const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
	    e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);// 信息矩阵 误差权重矩阵

	    g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
	    e12->setRobustKernel(rk1);// 核函数
	    rk1->setDelta(deltaHuber);
	    optimizer.addEdge(e12);
	    
       // 步骤4.2 ：添加 帧1 地图点 映射到 帧2特征点 的 边
	    // Set edge x2 = S21*X1
	    Eigen::Matrix<double,2,1> obs2;
	    const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
	    obs2 << kpUn2.pt.x, kpUn2.pt.y;// 帧2 特征点 的 实际值

	    g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

	    e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));// 帧1 地图点
	    e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));// 帧2 到 帧1 变换顶点
	    e21->setMeasurement(obs2);// 帧2 特征点 的 实际值
	    float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
	    e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

	    g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
	    e21->setRobustKernel(rk2);// 核函数
	    rk2->setDelta(deltaHuber);
	    optimizer.addEdge(e21);

	    vpEdges12.push_back(e12);
	    vpEdges21.push_back(e21);
	    vnIndexEdge.push_back(i);
	}
	

// 步骤5：g2o开始优化，先迭代5次	 
	optimizer.initializeOptimization();
	optimizer.optimize(5);


// 步骤6：剔除一些误差大的边
    // Check inliers
    // 进行卡方检验，大于阈值的边剔除
	int nBad=0;
	for(size_t i=0; i<vpEdges12.size();i++)
	{
	    g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
	    g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
	    if(!e12 || !e21)
		continue;

	    if(e12->chi2() > th2 || e21->chi2() > th2)// 误差较大 边噪声大
	    {
		size_t idx = vnIndexEdge[i];
		vpMatches1[idx]=static_cast<MapPoint*>(NULL);
		optimizer.removeEdge(e12);// 移除边
		optimizer.removeEdge(e21);
		vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
		vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
		nBad++;
	    }
	}

	int nMoreIterations;
	if(nBad > 0)
	    nMoreIterations=10;//除去外点后 再迭代10次
	else
	    nMoreIterations=5;

	if(nCorrespondences - nBad < 10)
	    return 0;
// 步骤7：再次g2o优化剔除误差大的边后剩下的边
	// Optimize again only with inliers
	optimizer.initializeOptimization();
	optimizer.optimize(nMoreIterations);
	
// 步骤8：再次进行卡方检验，统计 除去误差较大后的 内点个数
	int nIn = 0;
	for(size_t i=0; i<vpEdges12.size();i++)
	{
	    g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
	    g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
	    if(!e12 || !e21)
		continue;

	    if(e12->chi2()>th2 || e21->chi2()>th2)// 误差较大
	    {
		size_t idx = vnIndexEdge[i];
		vpMatches1[idx]=static_cast<MapPoint*>(NULL);
	    }
	    else
		nIn++;//内点个数
	}
	
// 步骤9：得到优化后的结果
	// Recover optimized Sim3
	g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
	g2oS12= vSim3_recov->estimate();

	return nIn;
    }


} //namespace ORB_SLAM
