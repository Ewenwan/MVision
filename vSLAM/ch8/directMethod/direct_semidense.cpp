/*
 * 半稠密直接法  求相机位姿
 * ./direct_sparse ../../data/
 *  * 空间点 P = (X, Y, Z) = (X, Y, Z, 1) 非齐次，齐次表示 (RGB-D获取3D坐标)
 * 两帧图像上 对应 的像素坐标 p1  p2 = (u, v) = (u, v, 1) 非齐次，齐次表示
 * 相机内参数 K
 * 第一帧到第二帧的运动 旋转矩阵 R  平移向量t  也即变换矩阵 T  4*4    李代数形式  exp(f)    4*4 
 * 
 * p1 = K * P / Z1 , Z1为 P点在 第一帧图像下的深度
 * p2 = K * (R*P + t) / Z2  = K * (T*P)前三列 / Z2 = K * exp(f)*P前三列  / Z2 , Z2为 P点在 第二帧图像下的深度
 * 
 * 特征法中 通过匹配特征点描述子，可知道配对的 p1，p2像素位置，可以计算2D-2D 2D-3D  3D-3D 重投影误差得到 R t
 * 而在 直接法  中 没有特征匹配，无从知道哪一个p2 与p1 匹配 (光流法中 通过灰度不变得到 匹配点对)
 * 
 * 直接法的思路是 根据当前相机的位姿估计值和p1  来寻找 p2 的位置 最小化的不是重投影误差
 * 而是 光度误差（灰度误差）与光流法一样 同一个点 灰度值相近(或不变)
 *  e = I(p1) - I2(p2) 
 *  有N个 空间点 Pi
 *  最小化 sum(ei转置 * ei)
 *  e(f) =  I(p1) - I2(p2) =  I(K * P / Z1) - I2(K * exp(f)*P前三列  / Z2)     ；  是 李代数形式 变换矩阵的 函数 
 *  e(f 左乘 扰动∇f)   =  I(K * P / Z1) - I2(K * exp(∇f)*exp(f)*P)前三列  / Z2)
 *  =  I(K * P / Z1) - I2(K * (1+∇f)exp(f)*P)前三列  / Z2)
 *  =  I(K * P / Z1) - I2(K *exp(f)*P/ Z2  +  K *∇f * exp(f)*P/ Z2)
 * 
 *  记  Q = ∇f * exp(f)*P为 P在扰动之后 位于第二帧 相机坐标系下 的坐标
 *        u = K * Q/ Z2 为 其像素坐标
 * 泰勒展开  f(x +∇x ) =   f(x)  +                                                                                                                  f'(x) * ∇x   链式求导法则
 * e(f 左乘 扰动∇f)  = I(K * P / Z1) - I2(K *exp(f)*P/ Z2  +  u) = I(K * P / Z1) - I2(K *exp(f)*P/ Z2) - I2对u偏导 * u对Q偏导 * Q对 ∇f 偏导 * ∇f
 * = e(f)  - I2对u偏导 * u对Q偏导 * Q对 ∇f 偏导 * ∇f
 * 注: I2对u偏导 为 u点处的 像素灰度梯度  
 *       u对Q偏导 为投影方程 关于相机坐标系下 的三维点导数 Q = (X, Y, Z)    u = K * Q/ Z 
		 u       [fx 0 cx         X/Z
		 v  =     0 fy cy  *    Y/Z
		 1           0 0  1]      1
		 利用第三行消去s(实际上就是 P'的深度) 
		 u = fx * X/Z + cx
		 v = fy * Y/Z + cy 
		 u 对Q的偏导数 = - [ u对X的偏导数 u对Y的偏导数 u对Z的偏导数;
                                                  v对X的偏导数 v对Y的偏导数  v对Z的偏导数] 
           =   - [ fx/Z   0        -fx * X/Z ^2                                                                                                                       
                     0       fy/Z   -fy * Y/Z ^2]
 *  Q对∇f的偏导数 = [ I  -Q叉乘矩阵] 3*6大小   平移在前  旋转在后
 *  = [ 1 0  0   0   Z   -Y
 *       0 1  0  -Z   0    X
 *       0 0  1  Y   -X   0]
 * 有向量 t = [ a1 a2 a3] 其
 * 叉乘矩阵 = [0  -a3  a2;
 *                     a3  0  -a1; 
 *                    -a2 a1  0 ]                   
    u对∇f的偏导数 = u对Q偏导 *    Q对∇f的偏导数  2*6 矩阵  与图像无关
 * =  两者相乘得到 
 * = - [fx/Z   0       -fx * X/Z ^2   -fx * X*Y/Z^2      fx + fx * X^2/Z^2    -fx*Y/Z
 *           0    fy/Z   -fy* Y/Z^2    -fy -fy* Y^2/Z^2   fy * X*Y'/Z^2          fy*X/Z   ] 
 * 如果是 旋转在前 平移在后 调换前三列  后三列 
 // 旋转在前 平移在后   g2o   负号乘了进去
 *=  [ fx *X*Y/Z^2           -fx *(1 + X^2/Z^2)   fx*Y/Z  -fx/Z   0        fx * X/Z^2 
 *      fy *(1 + Y^2/Z^2)  -fy * X*Y/Z^2           -fy*X/Z   0      -fy/Z   fy* Y/Z^2     ]                  
       
     得到 误差e 对李代数的导数  雅克比矩阵
     J = I2对u偏导 * u对 ∇f 偏导   
     // 前者计算 图像灰度梯度可以得到(x方向梯度 y方向梯度 离散求解 坐标前后灰度值作差/2)  后者按上式 得到   
 * 
 * 对于N个点的问题 可以使用优化问题的 雅克比 使用 高斯牛顿 GN或者 LM计算更新增量 迭代求解
 * 
 * 根据 空间点 P的来源 可以把直接法进行分类 
 * 【1】稀疏直接法        P来自 稀疏关键点  或者光流法跟踪的 关键点
 * 【2】半稠密直接法    J = I2对u偏导 * u对 ∇f 偏导  如果像素梯度为0 则J=0 没有左右 ，使用像素梯度不为0的点 实际可取x，y方向梯度平方和小于50 剔除
 * 【3】稠密直接法       P为 所有像素点   需要GPU加速

 */
#include <iostream>//输入输出流
#include <fstream>//文件流
#include <list>//列表
#include <vector>//容器
#include <chrono>//计时
#include <ctime>//时间
#include <climits>//限制
//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>//图像处理
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>//二维特征
//g2o 非线性凸优化
#include <g2o/core/base_unary_edge.h>//一元边
#include <g2o/core/block_solver.h>//矩阵快求解器 矩阵分解
#include <g2o/core/optimization_algorithm_levenberg.h>//LM 数字优化算法
#include <g2o/solvers/dense/linear_solver_dense.h>//线性方程求解器
#include <g2o/core/robust_kernel.h>//
#include <g2o/types/sba/types_six_dof_expmap.h>//系统定义的顶点类型   6自由度位姿

using namespace std;
using namespace g2o;

/********************************************
 * 本节演示了RGBD上的半稠密直接法 
 ********************************************/

// 一次测量的值，包括一个世界坐标系下三维点与一个灰度值
struct Measurement//结构体
{
    Measurement ( Eigen::Vector3d p, float g ) : pos_world ( p ), grayscale ( g ) {}//直接赋值初始化
    Eigen::Vector3d pos_world;//世界坐标系下三维点  双整数
    float grayscale;//点对应的灰度值   浮点型
};

// 2D坐标 由 相机内参数 和深度信息 得到 像极坐标系下的三维坐标
// 相机归一化 平面下坐标 x =  K逆* px  y =  K逆* py  相机坐标系下 归一化平面上的点 
 //相机内参数 K=
//   [fx 0 cx
//     0 fy cy
//     0 0  1]
// x= (px -cx)/fx
// y=(py-cy)/fy
// z = 1
// 而深度值 为 dd= d/scale   mm 转成m
// 则相机坐标系下的坐标为  xx = x*dd  yy = y * dd  zz = z * dd
inline Eigen::Vector3d project2Dto3D ( int x, int y, int d, float fx, float fy, float cx, float cy, float scale )
{
    float zz = float ( d ) /scale;
    float xx = zz* ( x-cx ) /fx;
    float yy = zz* ( y-cy ) /fy;
    return Eigen::Vector3d ( xx, yy, zz );
}

// 相机坐标系下 三维坐标 (x, y, z)转化成 像素坐标(u, v)
//  像素坐标 x = xx/zz * fx  + cx
//  像素坐标 y = yy/zz * fy  + cy 
inline Eigen::Vector2d project3Dto2D ( float x, float y, float z, float fx, float fy, float cx, float cy )
{
    float u = fx*x/z+cx;
    float v = fy*y/z+cy;
    return Eigen::Vector2d ( u,v );
}

// 直接法估计位姿
// 输入：测量值（空间点的灰度），新的灰度图，相机内参； 输出：相机位姿
// 返回：true为成功，false失败
bool poseEstimationDirect ( const vector<Measurement>& measurements, cv::Mat* gray, Eigen::Matrix3f& intrinsics, Eigen::Isometry3d& Tcw );

//g20图优化
// project a 3d point into an image plane, the error is photometric error
// an unary edge with one vertex SE3Expmap (the pose of camera)
// 边  误差 需要自己定义  直接法                                 测量值维度(灰度值)  数据类型   连接顶点类型
class EdgeSE3ProjectDirect: public BaseUnaryEdge< 1, double, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 类成员 有Eigen  变量时需要 显示 加此句话 宏定义

    EdgeSE3ProjectDirect() {}//默认构造函数
    //自定义构造函数，参数为:
    //一个3d点世界坐标系下坐标
    //内参矩阵的4个参数
    //参考图，灰度图
    EdgeSE3ProjectDirect ( Eigen::Vector3d point, float fx, float fy, float cx, float cy, cv::Mat* image )
        : x_world_ ( point ), fx_ ( fx ), fy_ ( fy ), cx_ ( cx ), cy_ ( cy ), image_ ( image )
    {}
//计算误差 覆写 计算误差的 虚函数
    virtual void computeError()
    {
        const VertexSE3Expmap* v  =static_cast<const VertexSE3Expmap*> ( _vertices[0] );//类型强转  相机位姿 
        Eigen::Vector3d x_local = v->estimate().map ( x_world_ );//第二帧下 相机坐标系下的三维坐标x_local[0]，x_local[1]，x_local[2]
	// 相机坐标系下 三维坐标 (x, y, z)转化成 像素坐标(u, v)
	
        float x = x_local[0]*fx_/x_local[2] + cx_;
        float y = x_local[1]*fy_/x_local[2] + cy_;
        // check x,y is in the image
        if ( x-4<0 || ( x+4 ) >image_->cols || ( y-4 ) <0 || ( y+4 ) >image_->rows )//像素点坐标超出 有效范围
        {
            _error ( 0,0 ) = 0.0;
            this->setLevel ( 1 );//边 效果差  不考虑
        }
        else
        {
	  // 这里 误差为e =   I(p2) - I2(p1)   原来e =  I(p1) - I2(p2)  所以 雅克比 相差一个 负号 
	  // 上面计算出来的  x，y 为浮点数形式  
	  // 需要得到 整数形式 的 坐标值 对应图像的亮度值 需要进行插值运算 这里使用了 双线性插值
            _error ( 0,0 ) = getPixelValue ( x,y ) - _measurement;//根据 像素坐标  和 灰度图得到 灰度值  - 测量值
        }
    }
    
    
//覆写求雅克比矩阵 的虚函数
    // plus in manifold
    virtual void linearizeOplus( )
    {
        if ( level() == 1 )
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }
        VertexSE3Expmap* vtx = static_cast<VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d xyz_trans = vtx->estimate().map ( x_world_ );   // q in book

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0/xyz_trans[2];
        double invz_2 = invz*invz;

        float u = x*fx_*invz + cx_;
        float v = y*fy_*invz + cy_;

        // jacobian from se3 to u,v
        // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
	 // 旋转在前 平移在后   g2o   u对∇f的偏导数  像素坐标 对 变换矩阵李代数增量 的导数 
	// J1=  [ fx *X*Y/Z^2           -fx *(1 + X^2/Z^2)   fx*Y/Z  -fx/Z   0        fx * X/Z^2 
	//         fy *(1 + Y^2/Z^2)  -fy * X*Y/Z^2           -fy*X/Z   0      -fy/Z   fy* Y/Z^2     ]   
	// 上面 误差为e =   I(p2) - I2(p1)   原来e =  I(p1) - I2(p2)  所以 雅克比 相差一个 负号 
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

        jacobian_uv_ksai ( 0,0 ) = - x*y*invz_2 *fx_;
        jacobian_uv_ksai ( 0,1 ) = ( 1+ ( x*x*invz_2 ) ) *fx_;
        jacobian_uv_ksai ( 0,2 ) = - y*invz *fx_;
        jacobian_uv_ksai ( 0,3 ) = invz *fx_;
        jacobian_uv_ksai ( 0,4 ) = 0;
        jacobian_uv_ksai ( 0,5 ) = -x*invz_2 *fx_;

        jacobian_uv_ksai ( 1,0 ) = - ( 1+y*y*invz_2 ) *fy_;
        jacobian_uv_ksai ( 1,1 ) = x*y*invz_2 *fy_;
        jacobian_uv_ksai ( 1,2 ) = x*invz *fy_;
        jacobian_uv_ksai ( 1,3 ) = 0;
        jacobian_uv_ksai ( 1,4 ) = invz *fy_;
        jacobian_uv_ksai ( 1,5 ) = -y*invz_2 *fy_;

	// I2对u偏导   J2   图像灰度梯度可以得到(x方向梯度 y方向梯度 离散求解 坐标前后灰度值作差/2) 
        //这里由于各个像素点其实是离散值，其实求的是差分，前一个像素灰度值减后一个像素灰度值，除以2，即认为是这个方向上的梯度
        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;
        jacobian_pixel_uv ( 0,0 ) = ( getPixelValue ( u+1,v )-getPixelValue ( u-1,v ) ) /2;//灰度梯度  x方向  离散形式
        jacobian_pixel_uv ( 0,1 ) = ( getPixelValue ( u,v+1 )-getPixelValue ( u,v-1 ) ) /2;// 灰度梯度 y方向
        _jacobianOplusXi = jacobian_pixel_uv*jacobian_uv_ksai;//最后的 雅克比矩阵 
    }

    // dummy read and write functions because we don't care...
    virtual bool read ( std::istream& in ) {}
    virtual bool write ( std::ostream& out ) const {}

protected://私有函数
    // get a gray scale value from reference image (bilinear interpolated)
// x，y 为浮点数形式  需要得到 整数形式 的 坐标值 对应图像的亮度值 需要进行插值运算 这里使用了 双线性插值
    inline float getPixelValue ( float x, float y )
    {
      //这里先说一下各个参数的类型：
        //image_为Mat*类型，图像指针，所以调用data时用->符号，
        //data为图像矩阵首地址，支持数组形式访问，data[]就是访问到像素的值了，此处为像素的灰度值，类型为uchar
        //关于step有点复杂，data[]中括号的式子有点复杂，总的意思就是y行乘上每行内存数，定位到行，然后在加上x，定位到像素
        //step具体解释在最后面有一些资料
        //image_->data[int(y)*image_->step + int(x)]这一步读到了x,y处的灰度值，类型为uchar，
        //但是后面由于线性插值，需要定位这个像素的位置，而不是他的灰度值，所以取其地址，赋值给data_ptr，记住它的位置，后面使用
      
        uchar* data_ptr = & image_->data[ int ( y ) * image_->step + int ( x ) ];//对应的 灰度图 的灰度值 的地址   行 *  step + 列 对应的 灰度值
        uchar* data = data_ptr ;//地址
        
         //由于x,y这里有可能带小数，但是像素位置肯定是整数，所以，问题来了，(1.2, 4.5)像素坐标处的灰度值为多少呢?OK,线性插值！
        //说一下floor(),std中的cmath函数。向下取整,返回不大于x的整数。例floor(4.9)=4
        //xx和yy，就是取到小数部分。例：x=4.9的话，xx=x-floor(x)就为0.9。y同理
        //    I(1.2, 4.5) 飞整数的像素值  为周围四点 的 二维线性插值   按距离四点距离大小为权重 
        //             1-xx       xx
        //   1-yy   I(1,4)    I(1,5)
        //    yy      I(2,4)    I(2,5)
        //
        float xx = x - floor ( x );// 计算出来的坐标的 小数部分  
        float yy = y - floor ( y );
        return float (
                   ( 1-xx ) * ( 1-yy ) * data[0] +
                   xx* ( 1-yy ) * data[1] +
                   ( 1-xx ) *yy*data[ image_->step ] +
                   xx*yy*data[image_->step+1]
               );
    }
public://公开变量
    Eigen::Vector3d x_world_;          // 3D point in world frame
    float cx_=0, cy_=0, fx_=0, fy_=0; // Camera intrinsics 相机内参
    cv::Mat* image_=nullptr;           // reference image  图像  image 灰度图
};

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"用法:  direct_semidense path_to_dataset"<<endl;
        return 1;
    }
    srand ( ( unsigned int ) time ( 0 ) );//随机数
    string path_to_dataset = argv[1];
    /*
     TMU数据集:
     rgb.txt       记录了RGB图像的采集时间 和对应的文件名
     depth.txt   记录了深度图像的采集时间 和对应的文件名
     /rgb           存放 rgb图像  png格式彩色图为八位3通道
     /depth       存放深度图像  深度图为 16位单通道图像
     groundtruth.txt 为外部运动捕捉系统采集到的相机位姿  time,tx,ty,tz,qx,qy,qz,qw
     RGB图像和 深度图像采集独立的 时间不同时，需要对数据进行一次时间上的对齐，
     时间间隔相差一个阈值认为是同一匹配图
     可使用 associate.py脚步完成   python associate.py rgb.txt   depth.txt  > associate.txt
     
     */
    string associate_file = path_to_dataset + "/associate.txt";

    ifstream fin ( associate_file );//

    string rgb_file, depth_file, time_rgb, time_depth;
    cv::Mat color, depth, gray;
    vector<Measurement> measurements;
    // 相机内参
    float cx = 325.5;
    float cy = 253.5;
    float fx = 518.0;
    float fy = 519.0;
    float depth_scale = 1000.0;
    //相机内参数 K
    //   [fx 0 cx
    //     0 fy cy
    //     0 0  1]
    Eigen::Matrix3f K;
    K<<fx,0.f,cx,0.f,fy,cy,0.f,0.f,1.0f;

    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();// 相机位姿 转换矩阵  相机位姿 [R t] 的齐次表示 4*4

    cv::Mat prev_color;
    // 我们以第一个图像为参考，对后续图像和参考图像做直接法
    for ( int index=0; index<10; index++ )
    {
        cout<<"*********** 循环 "<<index<<" ************"<<endl;
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        color = cv::imread ( path_to_dataset+"/"+rgb_file );
        depth = cv::imread ( path_to_dataset+"/"+depth_file, -1 );
        if ( color.data==nullptr || depth.data==nullptr )
            continue; 
        cv::cvtColor ( color, gray, cv::COLOR_BGR2GRAY );
        if ( index ==0 )//第一帧
        {
            // 选择灰度变化(梯度)明显的点 select the pixels with high gradiants 
	    // 而没有选择提取FAST特征点
	  /*
	   // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> keypoints;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect ( color, keypoints );//检测 特征点
            for ( auto kp:keypoints )
            {
                // 去掉邻近图像边缘处的点
                if ( kp.pt.x < 20 || kp.pt.y < 20 || ( kp.pt.x+20 ) >color.cols || ( kp.pt.y+20 ) >color.rows )
                    continue;//跳过以下
                ushort d = depth.ptr<ushort> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ];//对于特征点的深度
                if ( d==0 )
                    continue;//跳过
                Eigen::Vector3d p3d = project2Dto3D ( kp.pt.x, kp.pt.y, d, fx, fy, cx, cy, depth_scale );//2D像素坐标   转换成 相机坐标系下的 三维点 3D
                float grayscale = float ( gray.ptr<uchar> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ] );//特征点 对应的灰度值   坐标值为整数 需要取整
                measurements.push_back ( Measurement ( p3d, grayscale ) );//测量值为 三维点 和 对应图像的灰度值
            }
            prev_color = color.clone();//赋值 图像
            continue;//第一幅图 跳过 以下
	   */
            for ( int x=10; x<gray.cols-10; x++ )
                for ( int y=10; y<gray.rows-10; y++ )
                {
                    Eigen::Vector2d delta (
                        gray.ptr<uchar>(y)[x+1] - gray.ptr<uchar>(y)[x-1], //x方向梯度
                        gray.ptr<uchar>(y+1)[x] - gray.ptr<uchar>(y-1)[x]//有方向梯度
                    );
                    if ( delta.norm() < 50 )// 梯度平方和小于 50 放弃该点
                        continue;//放弃该点 
                    ushort d = depth.ptr<ushort> (y)[x];//对应的 深度值 
                    if ( d==0 )//深度为0 
                        continue;//放弃该点 
                    Eigen::Vector3d p3d = project2Dto3D ( x, y, d, fx, fy, cx, cy, depth_scale );// 实际3D点
                    float grayscale = float ( gray.ptr<uchar> (y) [x] );//灰度值
                    measurements.push_back ( Measurement ( p3d, grayscale ) );//得到测量值 
                }
            prev_color = color.clone();//复制 图像
            cout<<"add total "<<measurements.size()<<" measurements."<<endl;
            continue;
        }
        
        // 使用直接法计算相机运动
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时开始
        poseEstimationDirect ( measurements, &gray, K, Tcw );//测量值
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
        cout<<"直接法耗时 direct method costs time: "<<time_used.count() <<" seconds."<<endl;
        cout<<"转换矩阵 Tcw="<<Tcw.matrix() <<endl;
	

        // 画特征点 plot the feature points
        cv::Mat img_show ( color.rows*2, color.cols, CV_8UC3 );
        prev_color.copyTo ( img_show ( cv::Rect ( 0,0,color.cols, color.rows ) ) );
        color.copyTo ( img_show ( cv::Rect ( 0,color.rows,color.cols, color.rows ) ) );
        for ( Measurement m:measurements )
        {
            if ( rand() > RAND_MAX/5 )
                continue;
            Eigen::Vector3d p = m.pos_world;
            Eigen::Vector2d pixel_prev = project3Dto2D ( p ( 0,0 ), p ( 1,0 ), p ( 2,0 ), fx, fy, cx, cy );
            Eigen::Vector3d p2 = Tcw*m.pos_world;
            Eigen::Vector2d pixel_now = project3Dto2D ( p2 ( 0,0 ), p2 ( 1,0 ), p2 ( 2,0 ), fx, fy, cx, cy );
            if ( pixel_now(0,0)<0 || pixel_now(0,0)>=color.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=color.rows )
                continue;

            float b = 0;
            float g = 250;
            float r = 0;
            img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3] = b;
            img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3+1] = g;
            img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3+2] = r;
            
            img_show.ptr<uchar>( pixel_now(1,0)+color.rows )[int(pixel_now(0,0))*3] = b;
            img_show.ptr<uchar>( pixel_now(1,0)+color.rows )[int(pixel_now(0,0))*3+1] = g;
            img_show.ptr<uchar>( pixel_now(1,0)+color.rows )[int(pixel_now(0,0))*3+2] = r;
            cv::circle ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 4, cv::Scalar ( b,g,r ), 2 );
            cv::circle ( img_show, cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +color.rows ), 4, cv::Scalar ( b,g,r ), 2 );
        }
        cv::imshow ( "result", img_show );
        cv::waitKey ( 0 );

    }
    return 0;
}

bool poseEstimationDirect ( const vector< Measurement >& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw )
{
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;  // 求解的向量 顶点(姿态) 是6＊1的
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();
    DirectBlock* solver_ptr = new DirectBlock ( linearSolver );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N  高斯牛顿
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr ); // L-M                 
    g2o::SparseOptimizer optimizer;  
    optimizer.setAlgorithm ( solver );
    optimizer.setVerbose( true );

    // 添加顶点
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();//位姿
    pose->setEstimate ( g2o::SE3Quat ( Tcw.rotation(), Tcw.translation() ) );//旋转矩阵 和 平移向量
    pose->setId ( 0 );//id
    optimizer.addVertex ( pose );//添加顶点

    // 添加边
    int id=1;
    for ( Measurement m: measurements )
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect (
            m.pos_world,//3D 位置
            K ( 0,0 ), K ( 1,1 ), K ( 0,2 ), K ( 1,2 ), gray//相机内参数   灰度图
        );
        edge->setVertex ( 0, pose );//顶点
        edge->setMeasurement ( m.grayscale );//测量值为真是灰度值
        edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );//误差 权重 信息矩阵
        edge->setId ( id++ );
        optimizer.addEdge ( edge );
    }
    cout<<"边的数量 edges in graph: "<<optimizer.edges().size() <<endl;
    optimizer.initializeOptimization();//优化初始化
    optimizer.optimize ( 30 );//最大优化次数
    Tcw = pose->estimate();// 变换矩阵
}

