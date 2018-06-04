/**
 * BA Example 
 * Author: Xiang Gao
 * Date: 2016.3
 * Email: gaoxiang12@mails.tsinghua.edu.cn
 * 
　* 调用格式：命令 [第一个图] [第二个图]
 * 
 * 在这个程序中，我们读取两张图像，进行特征匹配。
  然后根据匹配得到的特征，计算相机运动以及特征点的位置。
  这是一个典型的Bundle Adjustment，我们用g2o进行优化。

将一个图像中的二维像素点根据相机内参数和深度信息投影到三维空间,
	再通过欧式变换关系[R t]变换到另一个图像的坐标系下，
	再通过相机内参数投影到其像素平面上。
	可以求的的误差，使用优化算法，更新[R t]来使得误差最小

	当把　三维点Pi　也作为优化参数时，可以这样考虑
	图像1 像素点 p = {p1,p2,p3,...,pn}  qi = [u,v]
	图像2 像素点 q = {q1,q2,q3,...,qn}
	物理空间3D点 P = {P1,P2,P3,...,Pn} 坐标系为图像1的坐标系 Pi = [X,Y,Z]
	相机内参数K = [fx, 0,  ux
		       0,  fy, uy
		       0,  0,  1]
	  K_inv = [f1/x, 0,  -ux
		   0,  1/fy, -uy
		   0,  0,    1] 

	则有：
	  3D点 Pi 投影到 图像1的像素坐标系下
	  K*Pi = fx*X + Z*ux   x1              d1*u'
		 fy*Y + Z*uy = y1 = d1 * pi' = d1*v'  d1为3D点在图像1下的深度
		 Z             z1              d1

		      u'
	   投影点pi'    v' =  1/d1 *  K*Pi  值　约接近　pi　误差越小
		       1

	3D点 Pi 投影到 图像2的像素坐标系下
	K*(R*Pi+t)=     x2               
			  y2 = d2 * qi'
			  z2
	  投影点　qi'  = 1/d2 *  K*(R*Pi+t)   d2为3D点在图像2下的深度

传统数学方法，可以消去Pi得到只关于　pi,qi,R,t的关系，可以使用对极几何和基础局E进行求解
        　理论上使用 8个匹配点对就可以求解
          可以使用RanSac 随机序列采样一致性方法获取更鲁棒的解　[R,t]


 最小二乘图优化方法　最小化误差求解
    使用 差平方和作为 误差函数：
       E = sum( (1/d1 *  K*Pi - [pi,1])^2 + (1/d2 *  K*(R*Pi+t) - [qi,1]^2) )
    求解Pi,R,t 使得　E最小
    它叫做最小化重投影误差问题（Minimization of Reprojection error）。
    在实际操作中，我们实际上是在调整每个　Pi，使得它们更符合每一次观测值pi和qi,
    也就是使每个误差项都尽量的小,
    由于这个原因，它也叫做捆集调整（Bundle Adjustment）。

    上述方程是一个非线性函数，该函数的额最优化问题是非凸的优化问题，

    求解E的极值，当E的导数为０时，取得

    那么如何使得　E的导数　E'=0呢?

    对　E'　进行 泰勒展开

      一阶泰勒展开　: E‘(x) =  E’(x0) + E’‘(x0)  * dx 
                    =  J  + H * dx = 0
                     dx = -H逆 * J转置 * E(x0)
                     也可以写成：
                     H * dx = -J转置 * E(x0)

    求解时，需要求得函数 E 对每一个优化变量的　偏导数形成偏导数矩阵(雅克比矩阵)J
    二阶偏导数求解麻烦使用一阶偏导数的平方近似代替
    H = J转置*J

    可以写成如下线性方程：
     J转置*J * dx = -J转置 * E(x0)
     这里　误差E(x0)可能会有不同的执行度　可以在其前面加一个权重　w
     J转置*J * dx = -J转置 * w * E(x0)

     A * dx = b    GS高斯牛顿优化算法

     (A + λI) = b   LM 莱文贝格－马夸特方法 优化算法

     Levenberg-Marquardt方法的好处就是在于可以调节
     如果下降太快，使用较小的λ，使之更接近高斯牛顿法
     如果下降太慢，使用较大的λ，使之更接近梯度下降法  Δ = − J转置 * F(X0)

     这里线性方程组的求解　多使用　矩阵分解的算法　常见 LU分解、LDLT分解和Cholesky分解、SVD奇异值分解等

     所以这里需要：
              1. 系数矩阵求解器，来求解　雅可比矩阵J　和　海塞矩阵H, BlockSolver；
              2. 数值优化算法　GS高斯牛顿优化算法/LM 莱文贝格－马夸特方法 优化算法
                         计算　 A /   (A + λI)   
              3. 线性方程求解器，从 PCG, CSparse, Choldmod中选

 */

// for std
#include <iostream>// 输入输出
// for opencv 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>//2d特征
// 
#include <boost/concept_check.hpp>
// for g2o
#include <g2o/core/sparse_optimizer.h>// 优化器　
#include <g2o/core/block_solver.h>//　稀疏矩阵求解器
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>// LM
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>// cholmod　线性方程组求解器
#include <g2o/solvers/csparse/linear_solver_csparse.h> // csparse 线性方程组求解器 
#include <g2o/solvers/pcg/linear_solver_pcg.h>// PCG 线性方程组求解器 
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>// pry + tx, ty, tz

// Eigen3 矩阵
#include <Eigen/Core>
#include <Eigen/Geometry>

// <chrono>是标准模板库中与时间有关的头文件
#include <chrono>
using namespace std;
using namespace cv; //opencv库命名空间
// 寻找两个图像中的对应点，像素坐标系
// 输入：img1, img2 两张图像
// 输出：points1, points2, 两组对应的2D点
int findCorrespondingPoints( const cv::Mat& img1, // 常亮对象的引用 图像
			     const cv::Mat& img2, 
			     vector<cv::Point2f>& points1, // 对象的引用 2d像素点坐标
			     vector<cv::Point2f>& points2 );

//g2o_BundleAdjustment 优化   计算旋转和平移
//使用 3D-2D点对  直接使用 深度图  可由 双目计算得到  或 RGB-D 结构光 飞行时间法
void bundleAdjustment (
    const vector<Point2f> points_r_2d,// 两幅图像 特征点对中  
    const vector<Point2f> points_l_2d// 特征点对中的 两一个 2D点
);

// 相机内参
//	相机内参数K = [fx, 0,  ux
//		       0,  fy, uy
//		       0,  0,  1]
double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;

int main( int argc, char** argv )
{
// 调用格式：命令 [第一个图] [第二个图]
    if (argc != 3)
    {
        cout<<"Usage: ba_example img1, img2"<<endl;
        exit(1);
    }
    
// 读取图像
    cv::Mat img1 = cv::imread( argv[1] ); 
    cv::Mat img2 = cv::imread( argv[2] ); 
    
// 找到对应点
    vector<cv::Point2f> pts1, pts2;
    if ( findCorrespondingPoints( img1, img2, pts1, pts2 ) == false )
    {
        cout<<"匹配点不够！"<<endl;
        return 0;
    }
    cout<<"找到了"<<pts1.size()<<"组对应特征点。"<<endl;
    bundleAdjustment(pts1, pts2);
// 构造g2o中的图
/*
// 初始化g2o-------------------------
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block; // pose 维度为 6 (优化变量维度),  landmark 维度为 3
    // 使用Cholmod中的线性方程求解器
    //g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new  g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> ();
    // CSparse 线性方程求解器
    //g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new  g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType> ();
    // 雅克比和海塞矩阵　求解器　6*3 的参数 六个优化变量 误差变量３维度　[u,v,1]-[u',v',1]
    //g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3( linearSolver );
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block ( linearSolver ); // 矩阵块求解器
    // 优化算法　L-M 下降   
    g2o::OptimizationAlgorithmLevenberg* algorithm_ptr = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );  
    // g2o::OptimizationAlgorithmGaussNewton* algorithm_ptr = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// 高斯牛顿
    // g2o::OptimizationAlgorithmDogleg* algorithm_ptr = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//狗腿算法 
    // 构造求解器
    g2o::SparseOptimizer    optimizer;
    optimizer.setAlgorithm( algorithm_ptr );
   // optimizer.setVerbose( false );

    cout<<" g2o::SparseOptimizer  初始化完成"<<endl;   
    // 添加节点
// 两个位姿节点
    for ( int i=0; i<2; i++ )
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if ( i == 0)
            v->setFixed( true ); // 第一个点固定为 I+0
        // 预设值为单位Pose，因为我们不知道任何信息
        v->setEstimate( g2o::SE3Quat() );// 初始值　四元素位姿　
        optimizer.addVertex( v );
    }

    cout<<" add two Vertex 两个相机位姿"<<endl;  
    // 很多个特征点的节点
    // 以第一帧为准
   int index = 2;// 优化 id
   // for ( size_t i=0; i<pts1.size(); i++ )
   for ( const cv::Point2f& p:pts1 )//范围for
    {
        g2o::VertexSBAPointXYZ* v1 = new g2o::VertexSBAPointXYZ();
        v1->setId( index++ );
        // 由于深度不知道，只能把深度设置为1了
        // 计算在图像1下的 3d点坐标 
        double z = 1.0;
        //double x = ( pts1[i].x - cx ) * z / fx; 
        //double y = ( pts1[i].y - cy ) * z / fy; 
        double x = ( p.x - cx ) * z / fx;
        double y = ( p.y - cy ) * z / fy; 
        // // g2o 中必须设置 marg
        v1->setEstimate( Eigen::Vector3d(x,y,z) );
        v1->setMarginalized(true);// 可以优化尺度 既优化后的3d坐标
        optimizer.addVertex( v1 );
    }

    cout<<" add 3d points Vertex 3d点"<<endl;  
    // 准备相机参数
    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );
    cout<<" add CameraParameters 相机参数"<<endl;    
    // 准备边　一边是xyz-3d点，一边是[u,v]像素坐标
    // 第一帧
    vector<g2o::EdgeProjectXYZ2UV*> edges;// 向量用来记录边　优化信息
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();//　边
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );// 3d点 顶点
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(0)) );//id为0 为图像1 第一帧 I+0　位姿
        edge->setMeasurement( Eigen::Vector2d(pts1[i].x, pts1[i].y ) );// 测量值　点通过边投影后的2d点
        edge->setInformation( Eigen::Matrix2d::Identity() );// 误差信息矩阵
        edge->setParameterId(0, 0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );//添加边
        edges.push_back(edge);//　记录边
    }
    // 第二帧
    for ( size_t i=0; i<pts2.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();//　边
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );// 3d点 顶点
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(1)) );//id为1  图像2第一帧
        edge->setMeasurement( Eigen::Vector2d(pts2[i].x, pts2[i].y ) );// 测量值　点通过边投影后的2d点
        edge->setInformation( Eigen::Matrix2d::Identity() );// 误差信息矩阵
        edge->setParameterId(0,0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }
    
    cout<<"开始优化"<<endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);//10次
    cout<<"优化完毕"<<endl;
    
    //我们比较关心两帧之间的变换矩阵
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
    Eigen::Isometry3d pose = v->estimate();//　欧式变换矩阵 
    cout<<"Pose="<<endl<<pose.matrix()<<endl;
    
    // 以及所有优化和的3d点的位置
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2));
        cout<<"vertex id "<<i+2<<", pos = ";
        Eigen::Vector3d pos = v->estimate();
        cout<<pos(0)<<","<<pos(1)<<","<<pos(2)<<endl;
    }
    
    // 估计inlier的个数
    int inliers = 0;
    for ( auto e:edges )
    {
        e->computeError();// 投影误差
        // chi2 就是 error*\Omega*error, 如果这个数很大，说明此边的值与其他边很不相符
        if ( e->chi2() > 1 )
        {
            cout<<"error = "<<e->chi2()<<endl;
        }
        else 
        {
            inliers++;
        }
    }
    
    cout<<"inliers in total points: "<<inliers<<"/"<<pts1.size() + pts2.size()<<endl;
    optimizer.save("ba.g2o");
*/
    return 0;
}


int findCorrespondingPoints( const cv::Mat& img1, 
                             const cv::Mat& img2, 
                             vector<cv::Point2f>& points1, 
                             vector<cv::Point2f>& points2 )
{
/*
    cv::ORB orb;//　orb特征点
    vector<cv::KeyPoint> kp1, kp2;//关键点 cv::Point2f
    cv::Mat desp1, desp2;//　orb特征点描述子
    orb( img1, cv::Mat(), kp1, desp1 );//得到关键点和描述子
    orb( img2, cv::Mat(), kp2, desp2 );
*/
    //--------------------第0部初始化------------------------------------------------------
    vector<cv::KeyPoint> kp1, kp2;//关键点 cv::Point2f
    cv::Mat desp1, desp2;//描述子
    //  OpenCV3 特征点检测器  描述子生成器 用法
    Ptr<FeatureDetector> ORB = ORB::create();         //特征点检测器    其他 BRISK   FREAK   
    //Ptr<DescriptorExtractor> descriptor = ORB::create();//描述子生成器
    // OpenCV2 特征点检测器  描述子生成器 用法
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );

    //------------------第一步:检测 Oriented FAST 角点位置-----------------------------
    ORB->detect ( img1, kp1 );
    ORB->detect ( img2, kp2 );

    //------------------第二步:根据角点位置计算 BRIEF 描述子-------------------------
    ORB->compute ( img1, kp1, desp1 );
    ORB->compute ( img2, kp2, desp2 );

    //------------------第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    cout<<"分别找到了"<<kp1.size()<<"和"<<kp2.size()<<"个特征点"<<endl;

    // 汉明二进制字符串相似度匹配   
 cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create( "BruteForce-Hamming");
//===== 对描述子进行匹配 使用 快速最近临　FLANN 匹配
// 鲁棒匹配器设置　描述子匹配器
    //Ptr<cv::flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1); //  LSH index parameters
    //Ptr<cv::flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);       //   flann search parameters
    // instantiate FlannBased matcher
    //Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
    double knn_match_ratio=0.8;// 最相似匹配和次相似匹配　需要的插值比值
    vector< vector<cv::DMatch> > matches_knn;//匹配子
    matcher->knnMatch( desp1, desp2, matches_knn, 2 );//钱两个最近的匹配
    vector< cv::DMatch > matches;
    for ( size_t i=0; i<matches_knn.size(); i++ )
    {
// 最相似的匹配距离　< 0.8　* 次相似匹配距离　为好的匹配　　无 模棱两可 的情况
        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
            matches.push_back( matches_knn[i][0] );// 好的匹配
    }
    
    if (matches.size() <= 20) //匹配点太少
        return false;
    
    for ( auto m:matches )
    {
        points1.push_back( kp1[m.queryIdx].pt );//记录匹配点
        points2.push_back( kp2[m.trainIdx].pt );
    }
    
    return true;
}

// g2o_BundleAdjustment 优化
void bundleAdjustment (
    const vector< Point2f > points_l_2d,// 两幅图像 特征点对中 其一 点  
    const vector< Point2f > points_r_2d// 特征点对中的 两一个 2D点
  )
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6 (优化变量维度),  landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );//优化算法
// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// 高斯牛顿
// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//狗腿算法
    g2o::SparseOptimizer optimizer;// 稀疏 优化模型
    optimizer.setAlgorithm ( solver ); // 设置求解器

    cout<<" g2o::SparseOptimizer  初始化完成"<<endl; 
    // 添加节点
    // 两个位姿节点    顶点1 vertex   优化变量 相机位姿
    for ( int i=0; i<2; i++ )
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap(); // camera pose   旋转矩阵 R   平移矩阵 t 的   李代数形式
        v->setId(i);
        if ( i == 0)
            v->setFixed( true ); // 第一个点固定为零
        // 预设值为单位Pose，因为我们不知道任何信息
        v->setEstimate( g2o::SE3Quat() );//
        optimizer.addVertex( v );
    }
    cout<<" add two Vertex 两个相机位姿"<<endl;  
    // 很多个特征点的节点
// 以第一帧为准
    // 顶点2 空间点
    int index = 2;// 优化 id
    for ( const Point2f p:points_l_2d )   // 
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();// 空间点
        point->setId ( index++ );// id ++ 
        // 由于深度不知道，只能把深度设置为1了
        // 计算在图像1下的 3d点坐标 
        double z = 1.0;
        //double x = ( pts1[i].x - cx ) * z / fx; 
        //double y = ( pts1[i].y - cy ) * z / fy; 
        double x = ( p.x - cx ) * z / fx;
        double y = ( p.y - cy ) * z / fy; 

        point->setEstimate ( Eigen::Vector3d ( x, y, z ) );
        point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex ( point );// 各个3维点 也是优化变量
    }

    cout<<" add 3d points Vertex 3d点"<<endl;  
    // parameter: camera intrinsics  相机内参数
        //相机内参数
    //   [fx 0 cx
    //     0 fy cy
    //     0 0  1]
    // 准备相机参数
    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );// fx, cx, cy, 0
    camera->setId ( 0 );
    optimizer.addParameter ( camera );//相机参数

    // 准备边　一边是xyz-3d点，一边是[u,v]像素坐标
    // 第一帧
    // 边  edges  误差项
    index = 0;
    for ( const Point2f p:points_l_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();// 3D - 2D 点对 误差项  投影方程
        edge->setId ( index );//  id 
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>( optimizer.vertex ( index + 2) ) );// 边 连接 的 顶点 其一 
        edge->setVertex ( 1, dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0) ));// 相机位姿 T
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );//观测数值测量值　点通过边投影后的2d点
        edge->setParameterId ( 0,0 );//参数
        edge->setInformation ( Eigen::Matrix2d::Identity() );//误差项系数矩阵  信息矩阵：单位阵协方差矩阵   横坐标误差  和 纵坐标 误差
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge ( edge );//添加边
        index++;
    }

 // 第二帧
    int index2 = 0;
    for ( const Point2f p:points_r_2d)
    {
        g2o::EdgeProjectXYZ2UV*  edge2 = new g2o::EdgeProjectXYZ2UV();//　边
        edge2->setId ( index + index2 );//  id 
        edge2->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index2+2)) );// 3d点 顶点
        edge2->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1)) );//id为1  图像2第一帧
        edge2->setMeasurement( Eigen::Vector2d(p.x, p.y ) );// 测量值　点通过边投影后的2d点
        edge2->setInformation( Eigen::Matrix2d::Identity() );// 误差信息矩阵
        edge2->setParameterId(0,0);
        // 核函数
        edge2->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge2 );
        //edges.push_back(edge);
        index2++;
    }


    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时开始
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization();//初始化
    optimizer.optimize ( 100 );//优化 次数
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
       //我们比较关心两帧之间的变换矩阵
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
    Eigen::Isometry3d pose = v->estimate();//　欧式变换矩阵 
    cout<< "Pose="<< endl << pose.matrix() << endl;

    // 以及所有优化和的3d点的位置
    for ( size_t i=0; i<points_l_2d.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2));
        cout<<"vertex id "<< i+2 <<", pos = ";
        Eigen::Vector3d pos = v->estimate();
        cout << pos(0)<<","<<pos(1)<<","<<pos(2) << endl;
    }
/*
     // 估计inlier的个数
    int inliers = 0;
    for ( auto e:edges )
    {
        e->computeError();// 投影误差
        // chi2 就是 error*\Omega*error, 如果这个数很大，说明此边的值与其他边很不相符
        if ( e->chi2() > 1 )
        {
            cout<<"error = "<<e->chi2()<<endl;
        }
        else 
        {
            inliers++;
        }
    }
    
    cout<<"inliers in total points: "<<inliers<<"/"<<pts1.size() + pts2.size()<<endl;
*/
}/**
 * BA Example 
 * Author: Xiang Gao
 * Date: 2016.3
 * Email: gaoxiang12@mails.tsinghua.edu.cn
 * 
　* 调用格式：命令 [第一个图] [第二个图]
 * 
 * 在这个程序中，我们读取两张图像，进行特征匹配。
  然后根据匹配得到的特征，计算相机运动以及特征点的位置。
  这是一个典型的Bundle Adjustment，我们用g2o进行优化。

将一个图像中的二维像素点根据相机内参数和深度信息投影到三维空间,
	再通过欧式变换关系[R t]变换到另一个图像的坐标系下，
	再通过相机内参数投影到其像素平面上。
	可以求的的误差，使用优化算法，更新[R t]来使得误差最小

	当把　三维点Pi　也作为优化参数时，可以这样考虑
	图像1 像素点 p = {p1,p2,p3,...,pn}  qi = [u,v]
	图像2 像素点 q = {q1,q2,q3,...,qn}
	物理空间3D点 P = {P1,P2,P3,...,Pn} 坐标系为图像1的坐标系 Pi = [X,Y,Z]
	相机内参数K = [fx, 0,  ux
		       0,  fy, uy
		       0,  0,  1]
	  K_inv = [f1/x, 0,  -ux
		   0,  1/fy, -uy
		   0,  0,    1] 

	则有：
	  3D点 Pi 投影到 图像1的像素坐标系下
	  K*Pi = fx*X + Z*ux   x1              d1*u'
		 fy*Y + Z*uy = y1 = d1 * pi' = d1*v'  d1为3D点在图像1下的深度
		 Z             z1              d1

		      u'
	   投影点pi'    v' =  1/d1 *  K*Pi  值　约接近　pi　误差越小
		       1

	3D点 Pi 投影到 图像2的像素坐标系下
	K*(R*Pi+t)=     x2               
			  y2 = d2 * qi'
			  z2
	  投影点　qi'  = 1/d2 *  K*(R*Pi+t)   d2为3D点在图像2下的深度

传统数学方法，可以消去Pi得到只关于　pi,qi,R,t的关系，可以使用对极几何和基础局E进行求解
        　理论上使用 8个匹配点对就可以求解
          可以使用RanSac 随机序列采样一致性方法获取更鲁棒的解　[R,t]


 最小二乘图优化方法　最小化误差求解
    使用 差平方和作为 误差函数：
       E = sum( (1/d1 *  K*Pi - [pi,1])^2 + (1/d2 *  K*(R*Pi+t) - [qi,1]^2) )
    求解Pi,R,t 使得　E最小
    它叫做最小化重投影误差问题（Minimization of Reprojection error）。
    在实际操作中，我们实际上是在调整每个　Pi，使得它们更符合每一次观测值pi和qi,
    也就是使每个误差项都尽量的小,
    由于这个原因，它也叫做捆集调整（Bundle Adjustment）。

    上述方程是一个非线性函数，该函数的额最优化问题是非凸的优化问题，

    求解E的极值，当E的导数为０时，取得

    那么如何使得　E的导数　E'=0呢?

    对　E'　进行 泰勒展开

      一阶泰勒展开　: E‘(x) =  E’(x0) + E’‘(x0)  * dx 
                    =  J  + H * dx = 0
                     dx = -H逆 * J转置 * E(x0)
                     也可以写成：
                     H * dx = -J转置 * E(x0)

    求解时，需要求得函数 E 对每一个优化变量的　偏导数形成偏导数矩阵(雅克比矩阵)J
    二阶偏导数求解麻烦使用一阶偏导数的平方近似代替
    H = J转置*J

    可以写成如下线性方程：
     J转置*J * dx = -J转置 * E(x0)
     这里　误差E(x0)可能会有不同的执行度　可以在其前面加一个权重　w
     J转置*J * dx = -J转置 * w * E(x0)

     A * dx = b    GS高斯牛顿优化算法

     (A + λI) = b   LM 莱文贝格－马夸特方法 优化算法

     Levenberg-Marquardt方法的好处就是在于可以调节
     如果下降太快，使用较小的λ，使之更接近高斯牛顿法
     如果下降太慢，使用较大的λ，使之更接近梯度下降法  Δ = − J转置 * F(X0)

     这里线性方程组的求解　多使用　矩阵分解的算法　常见 LU分解、LDLT分解和Cholesky分解、SVD奇异值分解等

     所以这里需要：
              1. 系数矩阵求解器，来求解　雅可比矩阵J　和　海塞矩阵H, BlockSolver；
              2. 数值优化算法　GS高斯牛顿优化算法/LM 莱文贝格－马夸特方法 优化算法
                         计算　 A /   (A + λI)   
              3. 线性方程求解器，从 PCG, CSparse, Choldmod中选

 */

// for std
#include <iostream>// 输入输出
// for opencv 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>//2d特征
// 
#include <boost/concept_check.hpp>
// for g2o
#include <g2o/core/sparse_optimizer.h>// 优化器　
#include <g2o/core/block_solver.h>//　稀疏矩阵求解器
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>// LM
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>// cholmod　线性方程组求解器
#include <g2o/solvers/csparse/linear_solver_csparse.h> // csparse 线性方程组求解器 
#include <g2o/solvers/pcg/linear_solver_pcg.h>// PCG 线性方程组求解器 
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>// pry + tx, ty, tz

// Eigen3 矩阵
#include <Eigen/Core>
#include <Eigen/Geometry>

// <chrono>是标准模板库中与时间有关的头文件
#include <chrono>
using namespace std;
using namespace cv; //opencv库命名空间
// 寻找两个图像中的对应点，像素坐标系
// 输入：img1, img2 两张图像
// 输出：points1, points2, 两组对应的2D点
int findCorrespondingPoints( const cv::Mat& img1, // 常亮对象的引用 图像
			     const cv::Mat& img2, 
			     vector<cv::Point2f>& points1, // 对象的引用 2d像素点坐标
			     vector<cv::Point2f>& points2 );

//g2o_BundleAdjustment 优化   计算旋转和平移
//使用 3D-2D点对  直接使用 深度图  可由 双目计算得到  或 RGB-D 结构光 飞行时间法
void bundleAdjustment (
    const vector<Point2f> points_r_2d,// 两幅图像 特征点对中  
    const vector<Point2f> points_l_2d// 特征点对中的 两一个 2D点
);

// 相机内参
//	相机内参数K = [fx, 0,  ux
//		       0,  fy, uy
//		       0,  0,  1]
double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;

int main( int argc, char** argv )
{
// 调用格式：命令 [第一个图] [第二个图]
    if (argc != 3)
    {
        cout<<"Usage: ba_example img1, img2"<<endl;
        exit(1);
    }
    
// 读取图像
    cv::Mat img1 = cv::imread( argv[1] ); 
    cv::Mat img2 = cv::imread( argv[2] ); 
    
// 找到对应点
    vector<cv::Point2f> pts1, pts2;
    if ( findCorrespondingPoints( img1, img2, pts1, pts2 ) == false )
    {
        cout<<"匹配点不够！"<<endl;
        return 0;
    }
    cout<<"找到了"<<pts1.size()<<"组对应特征点。"<<endl;
    bundleAdjustment(pts1, pts2);
// 构造g2o中的图
/*
// 初始化g2o-------------------------
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block; // pose 维度为 6 (优化变量维度),  landmark 维度为 3
    // 使用Cholmod中的线性方程求解器
    //g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new  g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> ();
    // CSparse 线性方程求解器
    //g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new  g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType> ();
    // 雅克比和海塞矩阵　求解器　6*3 的参数 六个优化变量 误差变量３维度　[u,v,1]-[u',v',1]
    //g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3( linearSolver );
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block ( linearSolver ); // 矩阵块求解器
    // 优化算法　L-M 下降   
    g2o::OptimizationAlgorithmLevenberg* algorithm_ptr = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );  
    // g2o::OptimizationAlgorithmGaussNewton* algorithm_ptr = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// 高斯牛顿
    // g2o::OptimizationAlgorithmDogleg* algorithm_ptr = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//狗腿算法 
    // 构造求解器
    g2o::SparseOptimizer    optimizer;
    optimizer.setAlgorithm( algorithm_ptr );
   // optimizer.setVerbose( false );

    cout<<" g2o::SparseOptimizer  初始化完成"<<endl;   
    // 添加节点
// 两个位姿节点
    for ( int i=0; i<2; i++ )
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if ( i == 0)
            v->setFixed( true ); // 第一个点固定为 I+0
        // 预设值为单位Pose，因为我们不知道任何信息
        v->setEstimate( g2o::SE3Quat() );// 初始值　四元素位姿　
        optimizer.addVertex( v );
    }

    cout<<" add two Vertex 两个相机位姿"<<endl;  
    // 很多个特征点的节点
    // 以第一帧为准
   int index = 2;// 优化 id
   // for ( size_t i=0; i<pts1.size(); i++ )
   for ( const cv::Point2f& p:pts1 )//范围for
    {
        g2o::VertexSBAPointXYZ* v1 = new g2o::VertexSBAPointXYZ();
        v1->setId( index++ );
        // 由于深度不知道，只能把深度设置为1了
        // 计算在图像1下的 3d点坐标 
        double z = 1.0;
        //double x = ( pts1[i].x - cx ) * z / fx; 
        //double y = ( pts1[i].y - cy ) * z / fy; 
        double x = ( p.x - cx ) * z / fx;
        double y = ( p.y - cy ) * z / fy; 
        // // g2o 中必须设置 marg
        v1->setEstimate( Eigen::Vector3d(x,y,z) );
        v1->setMarginalized(true);// 可以优化尺度 既优化后的3d坐标
        optimizer.addVertex( v1 );
    }

    cout<<" add 3d points Vertex 3d点"<<endl;  
    // 准备相机参数
    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );
    cout<<" add CameraParameters 相机参数"<<endl;    
    // 准备边　一边是xyz-3d点，一边是[u,v]像素坐标
    // 第一帧
    vector<g2o::EdgeProjectXYZ2UV*> edges;// 向量用来记录边　优化信息
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();//　边
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );// 3d点 顶点
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(0)) );//id为0 为图像1 第一帧 I+0　位姿
        edge->setMeasurement( Eigen::Vector2d(pts1[i].x, pts1[i].y ) );// 测量值　点通过边投影后的2d点
        edge->setInformation( Eigen::Matrix2d::Identity() );// 误差信息矩阵
        edge->setParameterId(0, 0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );//添加边
        edges.push_back(edge);//　记录边
    }
    // 第二帧
    for ( size_t i=0; i<pts2.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();//　边
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );// 3d点 顶点
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(1)) );//id为1  图像2第一帧
        edge->setMeasurement( Eigen::Vector2d(pts2[i].x, pts2[i].y ) );// 测量值　点通过边投影后的2d点
        edge->setInformation( Eigen::Matrix2d::Identity() );// 误差信息矩阵
        edge->setParameterId(0,0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }
    
    cout<<"开始优化"<<endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);//10次
    cout<<"优化完毕"<<endl;
    
    //我们比较关心两帧之间的变换矩阵
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
    Eigen::Isometry3d pose = v->estimate();//　欧式变换矩阵 
    cout<<"Pose="<<endl<<pose.matrix()<<endl;
    
    // 以及所有优化和的3d点的位置
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2));
        cout<<"vertex id "<<i+2<<", pos = ";
        Eigen::Vector3d pos = v->estimate();
        cout<<pos(0)<<","<<pos(1)<<","<<pos(2)<<endl;
    }
    
    // 估计inlier的个数
    int inliers = 0;
    for ( auto e:edges )
    {
        e->computeError();// 投影误差
        // chi2 就是 error*\Omega*error, 如果这个数很大，说明此边的值与其他边很不相符
        if ( e->chi2() > 1 )
        {
            cout<<"error = "<<e->chi2()<<endl;
        }
        else 
        {
            inliers++;
        }
    }
    
    cout<<"inliers in total points: "<<inliers<<"/"<<pts1.size() + pts2.size()<<endl;
    optimizer.save("ba.g2o");
*/
    return 0;
}


int findCorrespondingPoints( const cv::Mat& img1, 
                             const cv::Mat& img2, 
                             vector<cv::Point2f>& points1, 
                             vector<cv::Point2f>& points2 )
{
/*
    cv::ORB orb;//　orb特征点
    vector<cv::KeyPoint> kp1, kp2;//关键点 cv::Point2f
    cv::Mat desp1, desp2;//　orb特征点描述子
    orb( img1, cv::Mat(), kp1, desp1 );//得到关键点和描述子
    orb( img2, cv::Mat(), kp2, desp2 );
*/
    //--------------------第0部初始化------------------------------------------------------
    vector<cv::KeyPoint> kp1, kp2;//关键点 cv::Point2f
    cv::Mat desp1, desp2;//描述子
    //  OpenCV3 特征点检测器  描述子生成器 用法
    Ptr<FeatureDetector> ORB = ORB::create();         //特征点检测器    其他 BRISK   FREAK   
    //Ptr<DescriptorExtractor> descriptor = ORB::create();//描述子生成器
    // OpenCV2 特征点检测器  描述子生成器 用法
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );

    //------------------第一步:检测 Oriented FAST 角点位置-----------------------------
    ORB->detect ( img1, kp1 );
    ORB->detect ( img2, kp2 );

    //------------------第二步:根据角点位置计算 BRIEF 描述子-------------------------
    ORB->compute ( img1, kp1, desp1 );
    ORB->compute ( img2, kp2, desp2 );

    //------------------第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    cout<<"分别找到了"<<kp1.size()<<"和"<<kp2.size()<<"个特征点"<<endl;

    // 汉明二进制字符串相似度匹配   
 cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create( "BruteForce-Hamming");
//===== 对描述子进行匹配 使用 快速最近临　FLANN 匹配
// 鲁棒匹配器设置　描述子匹配器
    //Ptr<cv::flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1); //  LSH index parameters
    //Ptr<cv::flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);       //   flann search parameters
    // instantiate FlannBased matcher
    //Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
    double knn_match_ratio=0.8;// 最相似匹配和次相似匹配　需要的插值比值
    vector< vector<cv::DMatch> > matches_knn;//匹配子
    matcher->knnMatch( desp1, desp2, matches_knn, 2 );//钱两个最近的匹配
    vector< cv::DMatch > matches;
    for ( size_t i=0; i<matches_knn.size(); i++ )
    {
// 最相似的匹配距离　< 0.8　* 次相似匹配距离　为好的匹配　　无 模棱两可 的情况
        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
            matches.push_back( matches_knn[i][0] );// 好的匹配
    }
    
    if (matches.size() <= 20) //匹配点太少
        return false;
    
    for ( auto m:matches )
    {
        points1.push_back( kp1[m.queryIdx].pt );//记录匹配点
        points2.push_back( kp2[m.trainIdx].pt );
    }
    
    return true;
}

// g2o_BundleAdjustment 优化
void bundleAdjustment (
    const vector< Point2f > points_l_2d,// 两幅图像 特征点对中 其一 点  
    const vector< Point2f > points_r_2d// 特征点对中的 两一个 2D点
  )
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6 (优化变量维度),  landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );//优化算法
// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// 高斯牛顿
// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//狗腿算法
    g2o::SparseOptimizer optimizer;// 稀疏 优化模型
    optimizer.setAlgorithm ( solver ); // 设置求解器

    cout<<" g2o::SparseOptimizer  初始化完成"<<endl; 
    // 添加节点
    // 两个位姿节点    顶点1 vertex   优化变量 相机位姿
    for ( int i=0; i<2; i++ )
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap(); // camera pose   旋转矩阵 R   平移矩阵 t 的   李代数形式
        v->setId(i);
        if ( i == 0)
            v->setFixed( true ); // 第一个点固定为零
        // 预设值为单位Pose，因为我们不知道任何信息
        v->setEstimate( g2o::SE3Quat() );//
        optimizer.addVertex( v );
    }
    cout<<" add two Vertex 两个相机位姿"<<endl;  
    // 很多个特征点的节点
// 以第一帧为准
    // 顶点2 空间点
    int index = 2;// 优化 id
    for ( const Point2f p:points_l_2d )   // 
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();// 空间点
        point->setId ( index++ );// id ++ 
        // 由于深度不知道，只能把深度设置为1了
        // 计算在图像1下的 3d点坐标 
        double z = 1.0;
        //double x = ( pts1[i].x - cx ) * z / fx; 
        //double y = ( pts1[i].y - cy ) * z / fy; 
        double x = ( p.x - cx ) * z / fx;
        double y = ( p.y - cy ) * z / fy; 

        point->setEstimate ( Eigen::Vector3d ( x, y, z ) );
        point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex ( point );// 各个3维点 也是优化变量
    }

    cout<<" add 3d points Vertex 3d点"<<endl;  
    // parameter: camera intrinsics  相机内参数
        //相机内参数
    //   [fx 0 cx
    //     0 fy cy
    //     0 0  1]
    // 准备相机参数
    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );// fx, cx, cy, 0
    camera->setId ( 0 );
    optimizer.addParameter ( camera );//相机参数

    // 准备边　一边是xyz-3d点，一边是[u,v]像素坐标
    // 第一帧
    // 边  edges  误差项
    index = 0;
    for ( const Point2f p:points_l_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();// 3D - 2D 点对 误差项  投影方程
        edge->setId ( index );//  id 
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>( optimizer.vertex ( index + 2) ) );// 边 连接 的 顶点 其一 
        edge->setVertex ( 1, dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0) ));// 相机位姿 T
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );//观测数值测量值　点通过边投影后的2d点
        edge->setParameterId ( 0,0 );//参数
        edge->setInformation ( Eigen::Matrix2d::Identity() );//误差项系数矩阵  信息矩阵：单位阵协方差矩阵   横坐标误差  和 纵坐标 误差
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge ( edge );//添加边
        index++;
    }

 // 第二帧
    int index2 = 0;
    for ( const Point2f p:points_r_2d)
    {
        g2o::EdgeProjectXYZ2UV*  edge2 = new g2o::EdgeProjectXYZ2UV();//　边
        edge2->setId ( index + index2 );//  id 
        edge2->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index2+2)) );// 3d点 顶点
        edge2->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1)) );//id为1  图像2第一帧
        edge2->setMeasurement( Eigen::Vector2d(p.x, p.y ) );// 测量值　点通过边投影后的2d点
        edge2->setInformation( Eigen::Matrix2d::Identity() );// 误差信息矩阵
        edge2->setParameterId(0,0);
        // 核函数
        edge2->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge2 );
        //edges.push_back(edge);
        index2++;
    }


    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时开始
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization();//初始化
    optimizer.optimize ( 100 );//优化 次数
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
       //我们比较关心两帧之间的变换矩阵
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
    Eigen::Isometry3d pose = v->estimate();//　欧式变换矩阵 
    cout<< "Pose="<< endl << pose.matrix() << endl;

    // 以及所有优化和的3d点的位置
    for ( size_t i=0; i<points_l_2d.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2));
        cout<<"vertex id "<< i+2 <<", pos = ";
        Eigen::Vector3d pos = v->estimate();
        cout << pos(0)<<","<<pos(1)<<","<<pos(2) << endl;
    }
/*
     // 估计inlier的个数
    int inliers = 0;
    for ( auto e:edges )
    {
        e->computeError();// 投影误差
        // chi2 就是 error*\Omega*error, 如果这个数很大，说明此边的值与其他边很不相符
        if ( e->chi2() > 1 )
        {
            cout<<"error = "<<e->chi2()<<endl;
        }
        else 
        {
            inliers++;
        }
    }
    
    cout<<"inliers in total points: "<<inliers<<"/"<<pts1.size() + pts2.size()<<endl;
*/
}
