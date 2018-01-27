/**
* This file is part of ORB-SLAM2.
* 全局/局部 优化 使用G2O图优化
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
void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();// 地图的关键帧
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();// 地图的 地图点
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}

// BA 最小化重投影误差     关键帧   地图点   优化迭代次数
// 优化关键帧的位姿态 和  地图点坐标值
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());//地图点数量

    
// 【1】设置求解器类型 帧位姿 pose 维度为 6 (优化变量维度), 地图点  landmark 维度为 3
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;// pose 维度为 6 (优化变量维度),  landmark 维度为 3
 // typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6 (优化变量维度),  landmark 维度为 3
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();// 矩阵求解器 指针
    // linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
//【2】 设置求解器
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    
// 【3】 设置函数优化方法
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);// LM 莱马算法
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// 高斯牛顿
    // g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//狗腿算法

// 【4】设置稀疏优化求解器  
    g2o::SparseOptimizer optimizer;// 稀疏 优化模型
    optimizer.setAlgorithm(solver);// 设置求解器

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);// 优化停止标志

    long unsigned int maxKFid = 0;// 最大关键帧 ID

    // Set KeyFrame vertices 
//【5】添加位姿态顶点  设置 每一帧的 6自由度位姿 顶点
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];//图中的每一个关键帧
        if(pKF->isBad())//不好的帧 不优化  野帧
            continue;
	// 顶点 vertex   优化变量
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();//camera pose   旋转矩阵 R   平移矩阵 t 的   李代数形式
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose())); // 优化变量初始值  mat形式位姿 转成  SE3Quat 李代数形式
        vSE3->setId(pKF->mnId);// 顶点 id 
        vSE3->setFixed(pKF->mnId==0);// 初始帧 位姿固定为 单位对角矩阵 世界坐标原点
        optimizer.addVertex(vSE3);//添加顶点
        if(pKF->mnId > maxKFid)
            maxKFid = pKF->mnId;// 最大关键帧 ID
    }

    const float thHuber2D = sqrt(5.99);	 //  g2o 优化为 两个值 像素点坐标               时鲁棒优化系数
    const float thHuber3D = sqrt(7.815); //  g2o 优化为 3个值 像素点坐标 + 视差   时鲁棒优化系数

 //【5】添加3自由度 地图点顶点
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
	
  //【6】添加边 edge  地图点 和 各自观测帧 之间的 关系  SET EDGES
	// map<KeyFrame*,size_t>::const_iterator mit 
        for( auto mit=observations.begin(); mit!=observations.end(); mit++)
        {

            KeyFrame* pKF = mit->first;//观测到改点的一个关键帧
            if(pKF->isBad() || pKF->mnId > maxKFid)//这个关键帧 是野帧 或者 不在优化的顶点范围内 跳过
                continue;

            nEdges++;// 边 计数
            // 该地图点在 对应 观测帧 上  对应的 关键点 像素 坐标
            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];// 该观测帧对应的 改点在 图像上的 像素点坐标

   // 【7】对于单目相机 匹配点坐标 小于0 的话是单目
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
 /*  s*u       [fx 0 cx        X'
 *   s*v  =     0 fy cy  *     Y'
 *   s          0 0  1]        Z'
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
 *      0 1  0  -Z'  0    X'
 *      0 0  1  Y'   -X   0]
 * 有向量 t = [ a1 a2 a3] 其
 * 叉乘矩阵 = [0  -a3  a2;
 *            a3  0  -a1; 
 *            -a2 a1  0 ]  
 * 
 * 两者相乘得到 
 * J = - [fx/Z'   0      -fx * X'/Z' ^2   -fx * X'*Y'/Z' ^2      fx + fx * X'^2/Z' ^2    -fx*Y'/Z'
 *         0     fy/Z'   -fy* Y'/Z' ^2    -fy -fy* Y'^2/Z' ^2   fy * X'*Y'/Z' ^2          fy*X'/Z'    ] 
 * 如果是 旋转在前 平移在后 调换前三列  后三列 
 * 
 * [2]  优化 P点坐标值
 * e 对P的偏导数   = e 对P'的偏导数 *  P'对P的偏导数 = e 对P'的偏导数 * R
 * P' = R * P + t
 * P'对P的偏导数  = R
 * 
 */		
                if(bRobust)// 鲁棒优化
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);// 设置鲁棒优化
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
                    e->setRobustKernel(rk);// 鲁棒优化 
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
// 【10】 开始迭代优化
    // Optimize!
    optimizer.initializeOptimization();//初始化
    optimizer.optimize(nIterations);// 优化迭代

// Recover optimized data
// 【11】从优化结果恢复数据
    //Keyframes
    // 恢复关键帧   更新 帧位姿
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

    //Points
    // 恢复地图点
    // 更新地图点
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

// 仅仅优化普通帧 的位姿  地图点不优化
int Optimizer::PoseOptimization(Frame *pFrame)
{
// 【1】设置求解器类型 帧位姿 pose 维度为 6 (优化变量维度), 地图点  landmark 维度为 3
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    // linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
//【2】 设置求解器
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
// 【3】 设置函数优化方法
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    
 // 【4】设置稀疏优化求解器  
    g2o::SparseOptimizer optimizer;// 稀疏 优化模型
    optimizer.setAlgorithm(solver);
    int nInitialCorrespondences=0;

    // Set Frame vertex
 //【5】添加位姿态顶点  设置 每一帧的 6自由度位姿 顶点
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));//相机位姿顶点
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);// 优化器 添加 位姿 顶点

    // Set MapPoint vertices
    // 边
    const int N = pFrame->N;// 每个帧  的 地图点  个数
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;// 单目 边容器  保存边
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    // 边
    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;// 双目 / 深度 边容器 保存边
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);


    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)// 每个帧  的 地图点   
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];//该帧上 的 每一个地图点
        if(pMP)
        {
            // Monocular observation
// 【6】单目 添加仅 优化位姿 的 边 和 对应的  地图点(参数)
            if(pFrame->mvuRight[i]<0)// 匹配点坐标 小于0 的话是单目
            {
                nInitialCorrespondences++;//边的数量
                pFrame->mvbOutlier[i] = false;
              // 观测数据 像素点 坐标
                Eigen::Matrix<double,2,1> obs;// 观测数据 像素坐标
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];//关键点
                obs << kpUn.pt.x, kpUn.pt.y;//
             // 边 仅优化位姿   边 基础一元边  连接一个顶点
                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));//   连接一个顶点
                e->setMeasurement(obs);//测量值
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);//误差信息矩阵  权重

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);// 鲁棒优化
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
            
 // 【6】双目 添加仅 优化位姿 的 边 和 对应的  地图点(参数)           
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
                e->setRobustKernel(rk);// 鲁棒优化
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
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    // 优化10次

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));//初始值
        optimizer.initializeOptimization(0);//优化
        optimizer.optimize(its[it]);

        nBad=0;
// wyw  2018 添加	
	// 单目 更新外点标志
	if(pFrame->mvuRight[1]<0)// 匹配点坐标 小于0 的话是单目
	{	
		for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)// 单目 每一条边
		{
		    g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];//

		    const size_t idx = vnIndexEdgeMono[i];

		    if(pFrame->mvbOutlier[idx])//外点
		    {
			e->computeError();//计算该误差
		    }
		    const float chi2 = e->chi2();
		    if(chi2 > chi2Mono[it])
		    {                
			pFrame->mvbOutlier[idx]=true;// 确实是外点 不好的点
			e->setLevel(1);
			nBad++;
		    }
		    else
		    {
			pFrame->mvbOutlier[idx]=false;// 原来是外点  优化过后 误差变小了  变成内点了
			e->setLevel(0);
		    }

		    if(it==2)
			e->setRobustKernel(0);
		}
         }
        else{
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

		      if(it==2)
			  e->setRobustKernel(0);
		  }
        }
        if(optimizer.edges().size()<10)
            break;
  }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

// 本地 关键帧 优化
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{    
    // 本地关键帧  Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();// 返回全部序列 关键帧 容器
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // 本地地图点 Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    //固定关键帧  Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // 启动优化器  Setup optimizer
//【4】 求解器类型 
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
// 【5】迭代优化算法  
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
// 【6】设置优化器   
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

//【7】本地关键帧 位姿 顶点 Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;// 关键
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();//顶点类型
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));// 初始值
        vSE3->setId(pKFi->mnId);//id
        vSE3->setFixed(pKFi->mnId==0);// 第一帧关键帧 固定
        optimizer.addVertex(vSE3);// 添加顶点
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;// 本地关键帧 最大的
    }

 // 【8】设置固定关键帧顶点  Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);//固定 不变
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // 【9】设置 地图点 顶点 Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;// 地图点  单目
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;// 单目 边
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;// 双目地图点
    vpMapPointEdgeMono.reserve(nExpectedSize);// 

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;// 双目边
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;// 双目 关键帧 边
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;// 双目 地图点边
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();// 地图点 顶点类型
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));// 初始值
        int id = pMP->mnId+maxKFid+1;//id
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);// 添加顶点

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();// 地图点对应的 观测 帧

    // 【10】设置边Set edges
	 // map<KeyFrame*,size_t>::const_iterator mit
        for(auto  mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;// 每一个顶点的 观测关键帧帧 

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];// 观测帧 对应 该地图点 的 像素坐标

      // 【11】单目下 添加边 Monocular observation  单目观测下 观测值 像素坐标
                if(pKFi->mvuRight[mit->second]<0)
                {
		  // 观测值
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;
		  // 二元边
                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();// 二元边  两个顶点
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));// 地图点
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));// 帧 位姿
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);// 误差信息矩阵
		 // 鲁棒优化
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
	          // 参数信息 用于 雅克比矩阵    迭代优化
                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                
    // 【11】双目下 添加边  // Stereo observation 观测值 像素坐标和 视差
                else 
                {
		 // 观测值 像素坐标和 视差
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
		  // 二元边
                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);
		  // 鲁棒优化
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
                    optimizer.addEdge(e);
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
// 【12】 开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore)
    {

    // Check inlier observations
   // 更新内点 标志
//  if(pKFi->mvuRight[1]<0)
//  {      
	  for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
	  {
	      g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
	      MapPoint* pMP = vpMapPointEdgeMono[i];

	      if(pMP->isBad())
		  continue;
	      if(e->chi2()>5.991 || !e->isDepthPositive())
	      {
		  e->setLevel(1);
	      }
	      e->setRobustKernel(0);
	  }

	  for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
	  {
	      g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
	      MapPoint* pMP = vpMapPointEdgeStereo[i];

	      if(pMP->isBad())
		  continue;

	      if(e->chi2()>7.815 || !e->isDepthPositive())
	      {
		  e->setLevel(1);
	      }

	      e->setRobustKernel(0);
	  }

    // Optimize again without the outliers

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

	// Check inlier observations       
	for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
	{
	    g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
	    MapPoint* pMP = vpMapPointEdgeMono[i];

	    if(pMP->isBad())
		continue;

	    if(e->chi2()>5.991 || !e->isDepthPositive())
	    {
		KeyFrame* pKFi = vpEdgeKFMono[i];
		vToErase.push_back(make_pair(pKFi,pMP));
	    }
	}

	for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
	{
	    g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
	    MapPoint* pMP = vpMapPointEdgeStereo[i];

	    if(pMP->isBad())
		continue;

	    if(e->chi2()>7.815 || !e->isDepthPositive())
	    {
		KeyFrame* pKFi = vpEdgeKFStereo[i];
		vToErase.push_back(make_pair(pKFi,pMP));
	    }
	}

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    //优化后更新 关键帧 Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //优化后 更新地图点 Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

// 【】关键帧顶点 Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();// 顶点类型
        const int nIDi = pKF->mnId;
        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);// 闭环检测优化

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    // 闭环边
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // 普通边 Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
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

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}


} //namespace ORB_SLAM
