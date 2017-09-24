#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
// Eigen 几何模块
#include <Eigen/Geometry>

/****************************
* 本程序演示了 Eigen 几何模块的使用方法
****************************/

int main ( int argc, char** argv )
{  
  //注意一下类型名的最后一个字符为d表示双精度类型,换成f表示单精度类型，两种类型不能混用，必须显示转换
    // Eigen/Geometry 模块提供了各种旋转和平移的表示
    // 3D 旋转矩阵直接使用 Matrix3d 或 Matrix3f
 /****旋转向量****/
    // 旋转向量使用 AngleAxis, 它底层不直接是Matrix，但运算可以当作矩阵（因为重载了运算符）
    // 乘以该向量，表示进行一个坐标变换
    //任意旋转可用一个旋转轴和一个旋转角度来表示。
    //旋转向量，旋转向量的方向与旋转轴一致，长度为旋转角度。
     /*********************************/
    /*旋转向量　沿 Z 轴旋转 45 度         角度　轴 */
    Eigen::AngleAxisd rotation_vector ( M_PI/4, Eigen::Vector3d ( 0,0,1 ) );     //沿 Z 轴旋转 45 度
    cout .precision(3);
    cout<<"rotation matrix =\n"<<rotation_vector.matrix() <<endl;                //用matrix()转换成矩阵
    // 也可以直接赋值
    /*********************************/
   /*旋转矩阵*/
   Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();//单位阵
    rotation_matrix = rotation_vector.toRotationMatrix();//转成旋转矩阵　由罗德里格公式进行转换
    // 用 AngleAxis 可以进行坐标变换
    Eigen::Vector3d v ( 1,0,0 );
     /*************旋转向量进行坐标变换********************/
    Eigen::Vector3d v_rotated = rotation_vector * v;
    cout<<"(1,0,0) after rotation = "<<v_rotated.transpose()<<endl;
    // 或者用旋转矩阵
     /*****************旋转矩阵进行坐标变换****************/
    v_rotated = rotation_matrix * v;
    cout<<"(1,0,0) after rotation = "<<v_rotated.transpose()<<endl;

    /**欧拉角表示的旋转**/
    // 欧拉角: 可以将旋转矩阵直接转换成欧拉角
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles ( 2,1,0 ); // ZYX顺序，即roll pitch yaw顺序
    cout<<"yaw pitch roll = "<<euler_angles.transpose()<<endl;

   /***欧式变换矩阵表示旋转**/
    // 欧氏变换矩阵使用 Eigen::Isometry
    Eigen::Isometry3d T=Eigen::Isometry3d::Identity();// 虽然称为3d，实质上是4＊4的矩阵　　齐次坐标
    T.rotate ( rotation_vector );                                        // 按照rotation_vector进行旋转
    T.pretranslate ( Eigen::Vector3d ( 1,3,4 ) );               // 把平移向量设成(1,3,4)
    cout << "Transform matrix = \n" << T.matrix() <<endl;
    // 用变换矩阵进行坐标变换
    Eigen::Vector3d v_transformed = T*v;                              // 相当于R*v+t
    cout<<"(1,0,0) after Isometry3d tranformed = "<<v_transformed.transpose()<<endl;

    // 对于仿射和射影变换，使用 Eigen::Affine3d 和 Eigen::Projective3d 即可，略

    /*******四元数表示的旋转***********/
    // 可以直接把AngleAxis赋值给四元数，反之亦然 Quaterniond 表示双精度　四元素　Quaternionf　表示单精度四元素
    Eigen::Quaterniond q = Eigen::Quaterniond ( rotation_vector );// 表示沿Z 轴旋转 45 度　的四元素变换 
    cout<<"quaternion from AngleAxis rotation_vector = \n"<<q.coeffs() <<endl;   // 请注意coeffs的顺序是(x,y,z,w),w为实部，前三者为虚部
    // 也可以把旋转矩阵赋给它
    q = Eigen::Quaterniond ( rotation_matrix );
    cout<<"quaternion from rotation_matrix = \n"<<q.coeffs() <<endl;
    // 使用四元数旋转一个向量，使用重载的乘法即可
    /*注意程序表达形式和实际运算的不一样*/
    v_rotated = q*v; // 注意数学上是q*v*q^{-1}  而程序为了简化表示　直接使用　q*v代替
    cout<<"(1,0,0) after Quaterniond rotation = "<<v_rotated.transpose()<<endl;
  /*编程题目
   小萝卜１号位姿q1=[0.35,0.2,0.3,0.1],t1=[0.3,0.1,0.1]'　　　世界坐标系到相机变换
   小萝卜２号位姿q2=[-0.5,0.4,-0.1,0.2],t2=[-0.1,0.5,0.3]'
   小萝卜１号看到位于自身坐标系下p=[0.5,0,0.2]'
   求该向量在小萝卜２号下的坐标
   */  
  Eigen::Quaterniond q1(0.35,0.2,0.3,0.1);//wxyz q1.coeffs()  xyzw  q1.vec()  xyz
  //q1 << 0.35,0.2,0.3,0.1;
  Eigen::Matrix<double, 3, 1> t1;//float类型
  t1 <<  0.3,0.1,0.1;
  Eigen::Quaterniond q2(-0.5,0.4,-0.1,0.2);
  //q2 << -0.5,0.4,-0.1,0.2;
  Eigen::Matrix<double, 3, 1> t2;//float类型
  t2 << -0.1,0.5,0.3;
  Eigen::Matrix<double, 3, 1> p1;//float类型
  p1 << 0.5,0,0.2;
  
  cout<<"q1= \n"<< q1.coeffs() <<endl;
  cout<<"t1= \n"<< t1 <<endl;
  cout<<"q2= \n"<< q2.coeffs() <<endl;
  cout<<"t2= \n"<< t2 <<endl;

  /*
  q1.setIdentity(); 
  cout<<"q1 after setIdentity \n"<<q1.coeffs() <<endl;
   q2.setIdentity(); 
  cout<<"q2 after setIdentity \n"<<q2.coeffs() <<endl;
  */
  //规范化　　归一化   除以模长
   q1=q1.normalized();  
  cout<<"q1 after normalized\n"<<q1.coeffs() <<endl;
   q2=q2.normalized(); 
  cout<<"q2 after normalized \n"<<q2.coeffs() <<endl;
  
Eigen::Matrix3d q1rotation_matrix = Eigen::Matrix3d::Identity();//单位阵
q1rotation_matrix=q1.toRotationMatrix();
Eigen::Isometry3d Tc1w=Eigen::Isometry3d::Identity();// 虽然称为3d，实质上是4＊4的矩阵　　齐次坐标
  
Tc1w.rotate (q1rotation_matrix );                                    // 按照q1rotation_matrix进行旋转
Tc1w.pretranslate ( t1);                                                     // 把平移向量设成t1

//Eigen::Isometry3d Twc1=Tc1w.inverse();//由world 到c1的逆变换　　成　c1到world
Eigen::Matrix<double, 3, 1> pw=Tc1w.inverse()*p1;    //将c1坐标系下的点p1变换到world坐标系下

Eigen::Matrix3d q2rotation_matrix = Eigen::Matrix3d::Identity();//单位阵
q2rotation_matrix=q2.toRotationMatrix();
Eigen::Isometry3d Tc2w=Eigen::Isometry3d::Identity();// 虽然称为3d，实质上是4＊4的矩阵　　齐次坐标
  
Tc2w.rotate (q2rotation_matrix );                                    // 按照q1rotation_matrix进行旋转
Tc2w.pretranslate ( t2);                                                     // 把平移向量设成t1

Eigen::Matrix<double, 3, 1> p2=Tc2w*pw;    //将world坐标系下的点pw变换到c2坐标系下
cout<<"the loc of p1 in c1  = \n"<< p1<<endl;
cout<<"the loc of p1 in world  = \n"<< pw<<endl;
cout<<"the loc of p1 in c2 = \n"<< p2<<endl;
     

  return 0;
}
