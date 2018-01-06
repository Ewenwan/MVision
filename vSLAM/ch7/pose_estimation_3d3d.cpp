/*
 * 迭代最近点 Iterative Closest Point, ICP求解 3D坐标 到 3D坐标的转换矩阵(不用求解距离 激光SLAM 以及 RGB-D SLAM)
 * 使用 线性代数SVD奇异值分解 或者 非线性优化方法 求解
 * 
 * 使用深度图将 平面图  转化成 3维点
 * 特征点匹配  得到 
 * 三维点对 P={p1， p2, ... ,pn}   P' = {p1'， p2', ... ,pn'}
 * 存在 旋转矩阵R 和 平移矩阵t
 * 使得 pi = R* pi' + t
 * 
 * 当在激光SLAM直接使用 三维点进行 匹配(距离最近)得到3D点对(激光数据 特征不够丰富，匹配得到的点误匹配的多)
 * 求解方法为  ICP  Iterative Closet Point  迭代最近点
 * 
 * 【1】 线性代数求解  SVD奇异值分解方法
 * 误差    ei = pi - (R*pi' + t)
 * 最小化误差和 min (sum(ei^2))   ; e^2 =  (pi-p - R*(pi' - p'))^2 + (p - R*p' - t)^2得到 R t 
 *  (1) 计算两组 点集的质心位置 p  、 p‘ ，然后计算每个点的去质心坐标 qi = pi - p    qi’ = pi' - p‘
 *  (2)  min(sum(qi - R * qi')^2)  >>> 得到 旋转矩阵R  
 *  (3)  有R计算 t   t =  p - R * p'
 * 
 * (qi - R * qi')^2 = qi转置 * qi   - 2*qi转置 * R * qi  + qi转置 * R转置* R * qi  第一项与R无关 第三项 应为  R转置* R = I
 * 
 *  qi转置 * R * qi      W= sum(qi * qi转置) = U * 对角矩阵 * V转置   奇异值分解      R= U * V转置 
 * 
 *  【2】 非线性优化方法
 *  ei = pi -exp(f) * pi'  = P - P‘ 李代数形式 的 变换矩阵 对误差求导 得到 迭代优化 梯度
 *   
 * 3×6的雅克比矩阵  误差  对应的 导数  优化变量更新 增量
 * e 对 ∇f的导数  = P'对∇f的偏导数
*  P'对∇f的偏导数 = [ I  -P'叉乘矩阵] 3*6大小   平移在前  旋转在后
*  = [ 1 0  0  0   Z'   -Y' 
*      0 1  0  -Z' 0    X'
*      0 0  1  Y' -X   0]
* 旋转在前  平移在后
*  = [   0   Z'  -Y' 1 0  0 
*        -Z' 0    X' 0 1  0 
*        Y'  -X   0  0 0  1]
* 
* J = - P'对∇f的偏导数
*  = [   0   -Z'   Y'  -1  0  0 
*        Z'   0    -X'  0 -1  0 
*        -Y'  X’    0   0  0 -1]
	
 */
 
#include <iostream>//输入输出流
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// Eigen3 矩阵
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
// 非线性优化算法  图优化 G2O
#include <g2o/core/base_vertex.h>//顶点
#include <g2o/core/base_unary_edge.h>//边
#include <g2o/core/block_solver.h>//矩阵块 分解 求解器  矩阵空间 映射  分解
#include <g2o/core/optimization_algorithm_gauss_newton.h>//高斯牛顿法
#include <g2o/solvers/eigen/linear_solver_eigen.h>// 矩阵优化
#include <g2o/types/sba/types_six_dof_expmap.h>// 定义好的顶点类型 和 误差 变量更新算法  6维度 优化变量  例如 相机 位姿
#include <chrono>//算法计时

using namespace std;//标准库　命名空间
using namespace cv; //opencv库命名空间

//特征匹配 计算匹配点对
void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

//ICP求解 3D- 3D 点对 求解 R T 使用 线性代数SVD奇异值分解
void pose_estimation_3d3d (
    const vector<Point3f>& pts1,//3D
    const vector<Point3f>& pts2,//3D
    Mat& R, Mat& t//求解得到的 旋转矩阵 和 平移矩阵
);

//g2o_BundleAdjustment 优化   计算旋转和平移
//ICP算法求解3D-3D点对的 转换矩阵后使用  图优化 进行优化
void bundleAdjustment (
    const vector< Point3f >& pts1,//3D点
    const vector< Point3f >& pts2,//3D点
    Mat& R, Mat& t//初始R T 以及优化更新后 的 量
);

// 需自定义 边类型  误差项g2o edge
// 误差模型—— 曲线模型的边, 模板参数：观测值维度(输入的参数维度)，类型，连接顶点类型(优化变量系统定义好的顶点 或者自定义的顶点)
// 3D点—3D点的边
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>//基础一元 边类型
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;// 类成员 有Eigen  变量时需要 显示 加此句话 宏定义
    EdgeProjectXYZRGBDPoseOnly( const Eigen::Vector3d& point ) : _point(point) {}//直接赋值  观测值  3D点
   // 误差计算
    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );//0号顶点为 位姿    类型强转
        // measurement is p, point is p'
        _error = _measurement - pose->estimate().map( _point );//对观测点 进行 变换 后 与测量值做差 得到 误差
    }
    
   // 3d-3d自定义求解器 
    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);//0号顶点为 位姿    类型强转
        g2o::SE3Quat T(pose->estimate());//得到 变换矩阵
        Eigen::Vector3d xyz_trans = T.map(_point);//对点进行变换
        double x = xyz_trans[0];//变换后的 x  y  z
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        // 3×6的雅克比矩阵  误差  对应的 导数  优化变量更新 增量
	/*
	*  P'对∇f的偏导数 = [ I  -P'叉乘矩阵] 3*6大小   平移在前  旋转在后
	*  = [ 1 0  0   0   Z'   -Y' 
	*       0 1  0  -Z'  0    X'
	*       0 0  1  Y'   -X   0]
	* 旋转在前  平移在后
	*  = [   0   Z'   -Y' 1 0  0 
	*        -Z'  0    X'  0 1  0 
	*         Y'   -X   0  0 0  1]
	* 
	* J = - P'对∇f的偏导数
	*  = [   0   -Z'   Y'  -1  0  0 
	*         Z'   0    -X'  0 -1  0 
	*         -Y'  X’    0   0  0 -1]
	*/
        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;
        
        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;
        
        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }

    bool read ( istream& in ) {}
    bool write ( ostream& out ) const {}
    
protected:
    Eigen::Vector3d _point;
};

int main ( int argc, char** argv )
{
    if ( argc != 5 )
    {
        cout<<"用法: 。、pose_estimation_3d3d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );//彩色
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    // 关键点 匹配点对
    vector<KeyPoint> keypoints_1, keypoints_2;//关键点
    vector<DMatch> matches;//匹配点对
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    // 利用深度信息 对二位点对建立3D点对
    Mat depth1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    Mat depth2 = imread ( argv[4], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );// 相机内参,TUM Freiburg2
    vector<Point3f> pts1, pts2;
    for ( DMatch m:matches )
    {
      //深度
        ushort d1 = depth1.ptr<unsigned short> ( int ( keypoints_1[m.queryIdx].pt.y ) ) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        ushort d2 = depth2.ptr<unsigned short> ( int ( keypoints_2[m.trainIdx].pt.y ) ) [ int ( keypoints_2[m.trainIdx].pt.x ) ];
        if ( d1==0 || d2==0 )   // bad depth
            continue;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );//像素点转换成 相机坐标系下点
        Point2d p2 = pixel2cam ( keypoints_2[m.trainIdx].pt, K );//(x,y,1) 归一化坐标平面上的点
        float dd1 = float ( d1 ) /1000.0;//深度 尺度   原来单位为mm 变成 m
        float dd2 = float ( d2 ) /1000.0;
	//3D点
        pts1.push_back ( Point3f ( p1.x*dd1, p1.y*dd1, dd1 ) );//得到三维点
        pts2.push_back ( Point3f ( p2.x*dd2, p2.y*dd2, dd2 ) );
    }

    cout<<"3d-3d 点对数量: "<<pts1.size() <<endl;
    Mat R, t;
    cout<<"3d-3d 点对利用线性代数SVD奇异值分解求解变换矩阵"<<endl;
/*  (1) 计算两组 点集的质心位置 p  、 p‘ ，然后计算每个点的去质心坐标 qi = pi - p    qi’ = pi' - p‘
 *  (2)  min(sum(qi - R * qi')^2)  >>> 得到 旋转矩阵R    
 *  (3)  有R计算 t   t =  p - R * p'
 * 
 * (qi - R * qi')^2 = qi转置 * qi   - 2*qi‘转置 * R * qi  + qi‘转置 * R转置* R * qi’ 第一项与R无关 第三项 应为  R转置* R = I
 * 
 *  qi转置 * R * qi      W= sum(qi * qi’转置) = U * 对角矩阵 * V转置   奇异值分解      R= U * V转置 
 *   t =  p - R * p'
 */
    pose_estimation_3d3d ( pts1, pts2, R, t );
    cout<<"迭代最近点 ICP 通过 SVD 分解: "<<endl;
    cout<<"2-1 旋转矩阵 R = "<<R<<endl;//第二张图到第一张图的转换
    cout<<"平移矩阵 t = "<<t<<endl;
    cout<<"1-2 旋转矩阵 R_inv = "<<R.t() <<endl;//第一张图到第二张图的转换
    cout<<"t_inv = "<<-R.t() *t<<endl;

    cout<<"非线性优化方法  bundle adjustment 求解"<<endl;

    bundleAdjustment( pts1, pts2, R, t );
    
    // 验证 verify p1 = R*p2 + t
    for ( int i=0; i<5; i++ )
    {
        cout<<"p1 = "<<pts1[i]<<endl;
        cout<<"p2 = "<<pts2[i]<<endl;
        cout<<"(R*p2+t) = "<< 
            R * (Mat_<double>(3,1)<<pts2[i].x, pts2[i].y, pts2[i].z) + t
            <<endl;
        cout<<endl;
    }
}

// 计算特征点 找到匹配点对
void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3 
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2 
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
   // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

// 像素坐标变换到 相机 归一化平面上 u，v --->>>  (x,y,1)
Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

// 【1】 线性代数求解  SVD奇异值分解方法
/*  (1) 计算两组 点集的质心位置 p  、 p‘ ，然后计算每个点的去质心坐标 qi = pi - p    qi’ = pi' - p‘
 *  (2)  min(sum(qi - R * qi')^2)  >>> 得到 旋转矩阵R    
 *  (3)  有R计算 t   t =  p - R * p'
 * 
 * (qi - R * qi')^2 = qi转置 * qi   - 2*qi‘转置 * R * qi  + qi‘转置 * R转置* R * qi’ 第一项与R无关 第三项 应为  R转置* R = I
 * 
 *  qi转置 * R * qi      W= sum(qi * qi’转置) = U * 对角矩阵 * V转置   奇异值分解      R= U * V转置 
 *   t =  p - R * p'
 */
void pose_estimation_3d3d (
    const vector<Point3f>& pts1,//3D点容器
    const vector<Point3f>& pts2,
    Mat& R, Mat& t
)
{
  //【1】 求中心点
    Point3f p1, p2;     //三维点集的中心点  center of mass
    int N = pts1.size(); //点对数量
    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];//各维度求和
        p2 += pts2[i];
    }
    p1 = Point3f( Vec3f(p1) /  N);//求均值 得到中心点
    p2 = Point3f( Vec3f(p2) / N);
    // 【2】得到去中心坐标
    vector<Point3f>     q1 ( N ), q2 ( N ); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // 【3】计算需要进行奇异值分解的 W = sum(qi * qi’转置) compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3d ( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    cout<<"W="<<W<<endl;

    // 【4】对  W 进行SVD 奇异值分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout<<"U="<<U<<endl;
    cout<<"V="<<V<<endl;
   // 【5】计算旋转 和平移矩阵 R  和 t 
    //  R= U * V转置 
    Eigen::Matrix3d R_ = U* ( V.transpose() );
    // t =  p - R * p'
    Eigen::Vector3d t_ = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R_ * Eigen::Vector3d ( p2.x, p2.y, p2.z );

    // 【6】格式转换 convert to cv::Mat
    R = ( Mat_<double> ( 3,3 ) <<
          R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
          R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
          R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
        );
    t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );
}

void bundleAdjustment (
    const vector< Point3f >& pts1,
    const vector< Point3f >& pts2,
    Mat& R, Mat& t )
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block( linearSolver );      // 矩阵块求解器 矩阵分解
    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm( solver );

    // 顶点 vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); //位姿 camera pose
    pose->setId(0);
    pose->setEstimate( g2o::SE3Quat(
        Eigen::Matrix3d::Identity(),
        Eigen::Vector3d( 0,0,0 )
    ) );
    optimizer.addVertex( pose );//添加顶点

    // 边 edges
    int index = 1;
    vector<EdgeProjectXYZRGBDPoseOnly*> edges;//所有的 边
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly( 
            Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z) );//点2
        edge->setId( index );
        edge->setVertex( 0, dynamic_cast<g2o::VertexSE3Expmap*> (pose) );//顶点
        edge->setMeasurement( Eigen::Vector3d( 
            pts1[i].x, pts1[i].y, pts1[i].z) );// 点1 = T * 点2 = 点1'
        edge->setInformation( Eigen::Matrix3d::Identity()*1e4 );//误差项系数矩阵  信息矩阵
        optimizer.addEdge(edge);//向图中添加边
        index++;
        edges.push_back(edge);//所有的边
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时开始
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize(10);// 最大优化次数为10
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"optimization costs time: "<<time_used.count()<<" seconds."<<endl;

    cout<<endl<<"优化后的 R t :"<<endl;
    cout<<"变换矩阵 T = "<<endl<<Eigen::Isometry3d( pose->estimate() ).matrix()<<endl;
    
}
