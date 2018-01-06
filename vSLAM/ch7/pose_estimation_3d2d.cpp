/*Pnp+g2o_BundleAdjustment
 *单目相机 经过对极几何初始化(八点法 、单应矩阵)求解一些3D点后
 * 或则 双目相机 深度相机可以直接得到 三维点 不需要初始化得到最开始的三维点
 *可使用3D点-2D点对 使用 直接线性变换DLT(Direct Linear Transform) n点透视问题(PnP)  最小二乘最小化重投影误差   非线性优化算法G2O
 * 
 * 【1】直接线性变换DLT    6个 3D - 2D 点对 
 * 2D点  s1 * (u，v，1) = T * (X, Y, Z, 1)  = T * P 齐次表示  T = [R t] 为 3*4的矩阵
 * T = [t1 t2 t3 t4; t5 t6 t7 t8; t9 t10 t11 t12]=[T1; T2; T3]
 * 可得到 u = T1 × P/(T3 * P)   v =  T2 × P/(T3 * P) 
 * 可得 T3 * P *u - T1 * P =0  以及 T3 * P *v - T2 * P =0   每个3D - 2D 点对 可提供 两个约束
 * T有 12个变量 至少需要6个 3D - 2D 点对 
 * 求解的 T的 左大半部分 不一定满足旋转矩阵R的约束关系，得到 T后需要使用QR分解 使得得到的 T 满足 SE3
 * 相当于 对 求出的 T重新映射到 SE3流形上  矩阵空间重映射 
 * 
 * 【2】 PnP 求解   只利用3个 3D - 2D 点对   多余点对 无用 噪声影响下 方法无效
 *   用于估计初始相机位置  再 构建最小二乘优化问题 进行调整
 *  世界坐标系下     三个点  A  B   C 
 *  对应图像坐标系 三个点  a   b   c 
 *  利用相似三角形    余弦定理 得到 世界坐标系下     三个点  A  B   C  对应 的 相机坐标系下的 x y z
 *   及转化为 3D - 3D 点对
 * 
 * 【3】bundle adjustment  重投影误差  
 *  三维点  Pi = (Xi, Yi, Zi)   相机坐标 pi = (xi, yi, 1)  像素坐标 c = (ui, vi)  
 * 相机相对于  世界坐标系(第一帧图像相机) 的 旋转 平移矩阵 R t (变换矩阵 T) 的 李代数形式 f   李群形式为 exp(f)
 * si * pi = K * T * Pi = K * exp(f) * Pi      这里 exp(f) * Pi  为 4*1维的需要为齐次表示 需要转换为 非齐次表示
 * 重投影误差  e =  sum( pi - 1/si * K * exp(f) * Pi )^2  ；   K * exp(f) * Pi 为三维点的重投影坐标
 * 最小化重投影误差 得到 变换矩阵李代数形式 f  
 * 由于  pi 最后一个为1  误差约束e 为两个方程   而 f  为6个自由度  x1 x2 x3 x4 x5 x6
 * 最小二乘优化 用于最小化一个函数   e(x + ∇x) = e(x)  +  J * ∇x
 * 所以 雅克比矩阵 J 为 2*6的矩阵
 * 
 * 雅克比J的推导：
 * si * pi = K * T * Pi = K * exp(f) * Pi  = K * Pi'   Pi'为相机坐标系下的坐标  exp(f) * Pi  前三维 (Xi', Yi', Zi') 
 *  s*u       [fx 0 cx       X'
 *  s*v  =     0 fy cy  *    Y'
 *   s         0 0  1]       Z'
 *  利用第三行消去s(实际上就是 P'的深度) 
 *  u = fx * X'/Z' + cx
 *  v = fy * Y'/Z'  + cy 
 * 
 * [1]
 *  我们对 变换矩阵 T) 的 李代数形式 f 左乘 扰动量 ∇f
 *  误差e 对∇f的偏导数 =  e 对P'的偏导数 *  P'对∇f的偏导数
 * 
 * e 对P'的偏导数 = - [ u对X'的偏导数 u对Y'的偏导数 u对Z'的偏导数;
 *                     v对X'的偏导数 v对Y'的偏导数  v对Z'的偏导数]  = - [ fx/Z'   0        -fx * X'/Z' ^2 
 *                                                                      0       fy/Z'    -fy* Y'/Z' ^2]
 *  P'对∇f的偏导数 = [ I  -P'叉乘矩阵] 3*6大小   平移在前  旋转在后
 *  = [ 1 0  0   0   Z'   -Y' 
 *      0 1  0  -Z'  0    X'
 *      0 0  1   Y'  -X   0]
 * 有向量 t = [ a1 a2 a3] 其
 * 叉乘矩阵 = [0  -a3  a2;
 *            a3  0  -a1; 
 *           -a2  a1  0 ]  
 * 
 * 两者相乘得到  平移在前 旋转在后
 * J = - [fx/Z'   0      -fx * X'/Z'^2   -fx * X'*Y'/Z' ^2      fx + fx * X'^2/Z'^2    -fx*Y'/Z'
 *         0     fy/Z'   -fy* Y'/Z'^2    -fy -fy* Y'^2/Z'^2     fy * X'*Y'/Z'^2        fy*X'/Z'    ] 
 * 如果是 旋转在前 平移在后 调换前三列  后三列 
 // 旋转在前 平移在后   g2o 
 * J =  [ fx *X'*Y'/Z'^2       -fx *(1 + X'^2/Z'^2)   fx*Y'/Z'  -fx/Z'   0       fx * X'/Z'^2 
 *        fy *(1 + Y'^2/Z'^2)  -fy * X'*Y'/Z'^2       -fy*X'/Z'   0     -fy/Z'   fy* Y'/Z'^2     ] 
 
 * [2]  
 * e 对P的偏导数   = e 对P'的偏导数 *  P'对P的偏导数 = e 对P'的偏导数 * R
 * P' = R * P + t
 * P'对P的偏导数  = R
 * 
 *
 *
 * G2O使用
 * 顶点1 相机位姿 变换 T 类型为  g2o::VertexSE3Expmap 李代数位姿 顶点2 相机1 特征点由深度得到的 空间三维点P 类型为 g2o::VertexSBAPointXYZ 
 * 边图像1 由深度 得到的 三维点 经过 位姿T 重投影在 图像二上 与 图像2上相对应的 特征点 的坐标误差 g2o::EdgeProjectXYZ2UV 投影方程边
 * 以上类型 为G2O已经定义好的类型  在 g2o/types/sba/types_six_dof_expmap.h
 *  
 * 
 // g2o::VertexSE3Expmap 李代数位姿
 class G2O_TYPES_SBA_API VertexSE3Expmap : public BaseVertex<6, SE3Quat>{//第一个参数 6 表示6维优化变量 类型为SE3Quat 四元素 + 位移向量表示
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexSE3Expmap();

  bool read(std::istream& is);
  bool write(std::ostream& os) const;

  virtual void setToOriginImpl() {
    _estimate = SE3Quat();
  }

  virtual void oplusImpl(const double* update_)  {
    Eigen::Map<const Vector6d> update(update_);//更新量
    setEstimate(SE3Quat::exp(update)*estimate());// 左乘 更新量 指数映射 李代数上的 增量
  }
};
 * 
 //g2o::EdgeProjectXYZ2UV 投影方程边                  二元边
 class G2O_TYPES_SBA_API EdgeProjectXYZ2UV : public  BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>{
  //观测维度为2 空间点的像素坐标  类型 Vector2D 连接的两个顶点  6维位姿  和 三维点
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZ2UV();
    bool read(std::istream& is);
    bool write(std::ostream& os) const;

    void computeError()  {
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);// 6维位姿  顶点
      const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);// 三维点 顶点
      const CameraParameters * cam
        = static_cast<const CameraParameters *>(parameter(0));
      Vector2D obs(_measurement);//观测值 
      _error = obs - cam->cam_map(v1->estimate().map(v2->estimate()));
      // 位姿 对 三维点 映射 得到 重投影 后的像素点   
      // 误差 = 观测值 - 重投影 后的像素点
    }

    virtual void linearizeOplus();// 雅克比矩阵  优化求解析
    CameraParameters * _cam;
};

// 雅克比矩阵  优化求解析 
void EdgeProjectXYZ2UV::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);// 6维位姿  顶点
  SE3Quat T(vj->estimate());// 转化成 变换矩阵形式
  VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);// 三维点 顶点
  Vector3D xyz = vi->estimate();  // 三维点
  Vector3D xyz_trans = T.map(xyz);// 三维点 经过 变换矩阵 重投影后的坐标

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z;//  对应 Z‘^2  雅克比矩阵求解中 有

  const CameraParameters * cam = static_cast<const CameraParameters *>(parameter(0));

//   
  Matrix<double,2,3,Eigen::ColMajor> tmp;
  tmp(0,0) = cam->focal_length;// fx
  tmp(0,1) = 0;
  tmp(0,2) = -x/z*cam->focal_length;// -fx * X‘ / Z'

  tmp(1,0) = 0;
  tmp(1,1) = cam->focal_length;     // fy 默认相等 fy = fx
  tmp(1,2) = -y/z*cam->focal_length;// - fy * Y'  / Z'

//误差 对 空间点 P的导数 =  e 对P'的偏导数 * R   维度  2*3  =  2*3   *  3*3
//  * e 对P'的偏导数 = - [ u对X'的偏导数 u对Y'的偏导数 u对Z'的偏导数;
 *                     v对X'的偏导数 v对Y'的偏导数  v对Z'的偏导数]  = - [ fx/Z'   0        -fx * X'/Z' ^2 
 *                                                                      0       fy/Z'    -fy* Y'/Z' ^2]
  _jacobianOplusXi =  -1./z * tmp * T.rotation().toRotationMatrix();
// 误差 对 位姿 T 的 导数   平移在前 旋转在后
 * J = - [fx/Z'   0      -fx * X'/Z'^2   -fx * X'*Y'/Z' ^2      fx + fx * X'^2/Z'^2    -fx*Y'/Z'
 *         0     fy/Z'   -fy* Y'/Z'^2    -fy -fy* Y'^2/Z'^2     fy * X'*Y'/Z'^2        fy*X'/Z'    ] 


// 旋转在前 平移在后
 * J =  [ fx *X'*Y'/Z'^2       -fx *(1 + X'^2/Z'^2)   fx*Y'/Z'  -fx/Z'   0       fx * X'/Z'^2 
 *        fy *(1 + Y'^2/Z'^2)  -fy * X'*Y'/Z'^2       -fy*X'/Z'   0     -fy/Z'   fy* Y'/Z'^2     ] 
 
  _jacobianOplusXj(0,0) =  x*y/z_2 *cam->focal_length;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *cam->focal_length;
  _jacobianOplusXj(0,2) = y/z * cam->focal_length;
  _jacobianOplusXj(0,3) = -1./z *cam->focal_length;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *cam->focal_length;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *cam->focal_length;
  _jacobianOplusXj(1,1) = -x*y/z_2 *cam->focal_length;
  _jacobianOplusXj(1,2) = -x/z *cam->focal_length;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *cam->focal_length;
  _jacobianOplusXj(1,5) = y/z_2 *cam->focal_length;
}


 * 
 * 
 *
 */
#include <iostream>//输入输出流
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>//2D特征
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// Eigen3 矩阵
#include <Eigen/Core>
#include <Eigen/Geometry>
// 非线性优化算法  图优化 G2O
#include <g2o/core/base_vertex.h>//顶点
#include <g2o/core/base_unary_edge.h>//边
#include <g2o/core/block_solver.h>//矩阵块 分解 求解器  矩阵空间 映射  分解
#include <g2o/core/optimization_algorithm_levenberg.h>// LM  函数最小化 优化算法
#include <g2o/solvers/csparse/linear_solver_csparse.h>  // 空间曲面 线性优化 
#include <g2o/types/sba/types_six_dof_expmap.h>// 定义好的顶点类型  6维度 优化变量  例如 相机 位姿  3维空间点  投影边等

#include <chrono>//时间计时

using namespace std;//标准库　命名空间
using namespace cv; //opencv库命名空间

//特征匹配 计算匹配点对
void find_feature_matches (
    const Mat& img_1, const Mat& img_2, // & 为引用  直接使用 参数本身 不进行复制  节省时间
    std::vector<KeyPoint>& keypoints_1,// 
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );//keypoint Descriptors Match   描述子匹配
// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

//g2o_BundleAdjustment 优化   计算旋转和平移
//使用 3D-2D点对  直接使用 深度图  可由 双目计算得到  或 RGB-D 结构光 飞行时间法
void bundleAdjustment (
    const vector<Point3f> points_3d,// 两幅图像 特征点对中 其一 点 根据深度信息 得到的 世界坐标系下的3D点
    const vector<Point2f> points_2d,// 特征点对中的 两一个 2D点
    const Mat& K,
    Mat& R, Mat& t
);

int main ( int argc, char** argv )
{
    if ( argc != 5 )// 命令行参数 img1 img2 depth1 depth2
    {
        cout<<"用法: ./pose_estimation_3d2d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );//彩色图
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    // 找到 两幅彩色图 中的 特征匹配点对
    vector<KeyPoint> keypoints_1, keypoints_2;//关键点
    vector<DMatch> matches;//特征点匹配对
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    // 建立3D点
     //相机内参数
    //   [fx 0 cx
    //     0 fy cy
    //     0 0  1]
    Mat d1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );// 相机内参,TUM Freiburg2
    vector<Point3f> pts_3d;//3D点  第一幅图像中的 特征点 对应的 3维点
    vector<Point2f> pts_2d;//2D点  第二幅图像中的 特征点
    for ( DMatch m:matches )
    {
        ushort d = d1.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];//匹配点对 对应的深度
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/1000.0;//深度单位为 毫米 mm  转换为m  除去尺度因子
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );// 像素坐标转相机归一化坐标  x，y，1
        pts_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );// 3D点  第一幅图像中的 特征点 对应的 3维点
        pts_2d.push_back ( keypoints_2[m.trainIdx].pt );// 2D点  第二幅图像中的 特征点
    }

    cout<<"3d-2d 点对数 : "<<pts_3d.size() <<endl;
// 利用 PnP 求解初始解
//只利用3个 3D - 2D 点对
    Mat r, t;//得到 初始 旋转向量r 和  平移矩阵t
    solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;//旋转矩阵
    cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵 罗德里的公式 旋转向量 得到 旋转矩阵
    cout<<"初始旋转矩阵 R="<<endl<<R<<endl;
    cout<<"初始平移向量 t="<<endl<<t<<endl;

    cout<<"使用 bundle adjustment优化算法 对R和 T进行优化： "<<endl;
    bundleAdjustment ( pts_3d, pts_2d, K, R, t );
}

//特征匹配 计算匹配点对 函数
void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //--------------------第0步:初始化------------------------------------------------------
    Mat descriptors_1, descriptors_2;//描述子
    //  OpenCV3 特征点检测器  描述子生成器 用法
    Ptr<FeatureDetector> detector = ORB::create();//特征点检测器    其他 BRISK   FREAK   
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // OpenCV2 特征点检测器  描述子生成器 用法
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );//二进制描述子 汉明点对匹配
    
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

// 像素坐标转相机归一化坐标
// 像素坐标转相机归一化坐标    x1 =  K逆* p1  x2 =  K逆* p2  相机坐标系下 归一化平面上的点 
    //相机内参数
    //   [fx 0 cx
    //     0 fy cy
    //     0 0  1]
Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),// x= (px -cx)/fx
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )//  y=(py-cy)/fy
           );
}

// g2o_BundleAdjustment 优化
void bundleAdjustment (
    const vector< Point3f > points_3d,// 两幅图像 特征点对中 其一 点 根据深度信息 得到的 世界坐标系下的3D点
    const vector< Point2f > points_2d,// 特征点对中的 两一个 2D点
    const Mat& K,
    Mat& R, Mat& t )
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

    // 顶点1 vertex   优化变量 相机位姿
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose   旋转矩阵 R   平移矩阵 t 的   李代数形式
    Eigen::Matrix3d R_mat;// 3 * 3 矩阵
    R_mat <<
               R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
               R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
               R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setId ( 0 );//id 优化 次数
    // 优化变量初始值
    pose->setEstimate ( g2o::SE3Quat (
                            R_mat,//旋转矩阵
                            Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )//平移矩阵
                        ) );
    optimizer.addVertex ( pose );//添加顶点
    // 顶点2 空间点
    int index = 1;// 优化 id
    for ( const Point3f p:points_3d )   // 3D 点  landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();// 空间点
        point->setId ( index++ );// id ++ 
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex ( point );// 各个3维点 也是优化变量
    }

    // parameter: camera intrinsics  相机内参数
        //相机内参数
    //   [fx 0 cx
    //     0 fy cy
    //     0 0  1]
    g2o::CameraParameters* camera = new g2o::CameraParameters (// jacoban need
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0 );// fx, cx, cy, 0
    camera->setId ( 0 );
    optimizer.addParameter ( camera );//相机参数

    // 边  edges  误差项
    index = 1;
    for ( const Point2f p:points_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();// 3D - 2D 点对 误差项  投影方程
        edge->setId ( index );//  id 
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );// 边 连接 的 顶点 其一 
        edge->setVertex ( 1, pose );// 相机位姿 T
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );//观测数值
        edge->setParameterId ( 0,0 );//参数
        edge->setInformation ( Eigen::Matrix2d::Identity() );//误差项系数矩阵  信息矩阵：单位阵协方差矩阵   横坐标误差  和 纵坐标 误差
        optimizer.addEdge ( edge );//添加边
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时开始
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization();//初始化
    optimizer.optimize ( 100 );//优化 次数
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
}
