#include <iostream>
#include <cmath>
using namespace std; 

#include <Eigen/Core>
#include <Eigen/Geometry>

// 李群李代数 库 
#include "sophus/so3.h"
#include "sophus/se3.h"
/*
 * sophus 库安装 
 * 本库为来版本 非模板的版本
 * git clone https://github.com//strasdat/Sophus.git
 * git checkout a621ff   版本
 * 在cmake编译
 * 
 SO(n) 特殊正交群      对应 n*n 的旋转矩阵  群(集合)
 SE(n+1) 特殊欧式群   对应   n*n 的旋转矩阵和 n*1的平移向量 组合成的  变换矩阵 群(集合)
 so(n)  对应 的李代数 为 so(n)   n×1  列向量   使得向量 和 代数 一一对应  可以使用代数的更新方法来更新 矩阵
 
 
 SO(3) 表示三维空间的 旋转矩阵 集合  3×3
 SE(3)  表示三维空间的 变换矩阵 集合  4×4
 李代数 so3的本质就是个三维向量，直接Eigen::Vector3d定义。 3个旋转
 李代数 se3的本质就是个六维向量，3个旋转 + 3个平移
 
 欧拉角定义：
 旋转向量定义的 李群SO(3)   Sophus::SO3 SO3_v( 0, 0, M_PI/2 );  // 亦可从旋转向量构造  这里注意，不是旋转向量的三个坐标值，有点像欧拉角构造。
                  旋转向量 转 旋转矩阵  Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
 旋转矩阵定义的 李群SO(3)   Sophus::SO3 SO3_R(R);               // Sophus::SO(3)可以直接从旋转矩阵构造
                  旋转矩阵 转 Eigen::Quaterniond q(R);            // 或者四元数(从旋转矩阵构造)
 四元素定义的     李群SO(3)   Sophus::SO3 SO3_q( q );
 李代数so3           为李群SO(3)  的对数映射  Eigen::Vector3d so3 = SO3_R.log();
 
  平移   Eigen::Vector3d t(1,0,0);           // 沿X轴平移1
  从旋转矩阵 和 平移t 构造  SE3      Sophus::SE3 SE3_Rt(R, t);           // 从R,t构造SE(3)
  从四元素     和 平移t 构造  SE3      Sophus::SE3 SE3_qt(q,t);            // 从q,t构造SE(3)
  李代数se(3) 是一个6维向量   为李群SE3 的对数映射
  typedef Eigen::Matrix<double,6,1> Vector6d;// Vector6d指代　Eigen::Matrix<double,6,1>
  Vector6d se3 = SE3_Rt.log();
 */



int main( int argc, char** argv )
{
 /*******************************************/
  // 旋转矩阵群/////特殊正交群　　　仅有　旋转没有平移
    // 沿Z轴转90度的旋转矩阵
    // 　　　　　　　　　　　旋转向量　　角度　　轴　　　　罗德里格公式进行转换为旋转矩阵　　　
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
    cout<<"RotationMatrix R: \n"<<R<<endl;
    
    /***李群*****/
    Sophus::SO3 SO3_R(R);               // Sophus::SO(3)可以直接从旋转矩阵构造
    Sophus::SO3 SO3_v( 0, 0, M_PI/2 );  // 亦可从旋转向量构造  这里注意，不是旋转向量的三个坐标值，有点像欧拉角构造。
    /*
     SO3  ::SO3(double rot_x, double rot_y, double rot_z)  
	  {  
	    unit_quaternion_  
		= (SO3::exp(Vector3d(rot_x, 0.f, 0.f))  
		  *SO3::exp(Vector3d(0.f, rot_y, 0.f))  
		  *SO3::exp(Vector3d(0.f, 0.f, rot_z))).unit_quaternion_;  
	  }
	显示的貌似是三个过程，先转Ｘ轴，再转Ｙ轴，再转Ｚ轴，完全跟旋转向量不搭边。瞅着过程有点像欧拉角的过程，三个轴分了三步。  
	我就有一个(1, 1, 1)旋转向量，如何构造成SO3呢？也就是让它输出(1, 1, 1)。
        Eigen::Vector3d so33 (1, 1, 1);  
        Sophus::SO3 SO3 =Sophus::SO3::exp(so33);  //李代数 指数映射成 旋转矩阵 对于的 李群
         cout<<"SO3=\n"<<SO3<<endl;  
     */
    Eigen::Quaterniond q(R);            // 或者四元数(从旋转矩阵构造)
    Sophus::SO3 SO3_q( q );
    // 上述表达方式都是等价的
    // 输出SO(3)时，以so(3)形式输出
    //从输出的形式可以看出，虽然SO3是李群，是旋转矩阵，但是输出形式还是向量（被转化成李代数输出）。
    // 重载了 << 运算符  out_str << so3.log().transpose() << std::endl;  
    cout<<"SO(3) from matrix: "<<SO3_R<<endl;  //SO(3) from matrix:      0      0 1.5708  
    cout<<"SO(3) from vector: "<<SO3_v<<endl;
    cout<<"SO(3) from quaternion :"<<SO3_q<<endl;
    
    /****李代数*****/
    // 使用对数映射获得它的李代数
    // 所以，李代数 so3的本质就是个三维向量，直接Eigen::Vector3d定义。
    Eigen::Vector3d so3 = SO3_R.log();
    cout<<"so3 = "<<so3.transpose()<<endl;
    // hat 为向量 到反对称矩阵  相当于　^　运算。
    cout<<"so3 hat=\n"<<Sophus::SO3::hat(so3)<<endl;
    // 相对的，vee为 反对称矩阵 到 向量  相当于下尖尖运算 
    cout<<"so3 hat vee= "<<Sophus::SO3::vee( Sophus::SO3::hat(so3) ).transpose()<<endl; // transpose纯粹是为了输出美观一些
    
    /****李代数求导　更新*****/
    // 增量扰动模型的更新
    Eigen::Vector3d update_so3(1e-4, 0, 0); //假设更新量为这么多
    Sophus::SO3 SO3_updated = Sophus::SO3::exp(update_so3)*SO3_R;// 增量指数映射×原李代数
    cout<<"SO3 updated = "<<SO3_updated<<endl;
    
    
    /********************萌萌的分割线*****************************/
    /************特殊欧式群　变换矩阵群　有旋转有平移*********************/
    cout<<"************我是分割线*************"<<endl;
    // 李群 对SE(3)操作大同小异
    Eigen::Vector3d t(1,0,0);           // 沿X轴平移1
    Sophus::SE3 SE3_Rt(R, t);           // 从R,t构造SE(3)
    Sophus::SE3 SE3_qt(q,t);            // 从q,t构造SE(3)
    cout<<"SE3 from R,t= "<<endl<<SE3_Rt<<endl;
    cout<<"SE3 from q,t= "<<endl<<SE3_qt<<endl;
    // 李代数se(3) 是一个六维向量，方便起见先typedef一下
    typedef Eigen::Matrix<double,6,1> Vector6d;// Vector6d指代　Eigen::Matrix<double,6,1>
    Vector6d se3 = SE3_Rt.log();
    cout<<"se3 = "<<se3.transpose()<<endl;
    // 观察输出，会发现在Sophus中，se(3)的平移在前，旋转在后.
    // 同样的，有hat和vee两个算符
    cout<<"se3 hat = "<<endl<<Sophus::SE3::hat(se3)<<endl;
    cout<<"se3 hat vee = "<<Sophus::SE3::vee( Sophus::SE3::hat(se3) ).transpose()<<endl;
    
    // 最后，演示一下更新
    Vector6d update_se3; //更新量
    update_se3.setZero();
    update_se3(0,0) = 1e-4d;
    Sophus::SE3 SE3_updated = Sophus::SE3::exp(update_se3)*SE3_Rt;
    cout<<"SE3 updated = "<<endl<<SE3_updated.matrix()<<endl;
    
    return 0;
}
