#include <iostream>
using namespace std;
#include <ctime>
// Eigen 部分
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Dense>

#define MATRIX_SIZE 50

/****************************
* 本程序演示了 Eigen 基本类型的使用
****************************/

int main( int argc, char** argv )
{
   // 矩阵　Eigen::Matrix<float,行,列>　
    // Eigen 中所有向量和矩阵都是Eigen::Matrix，它是一个模板类。它的前三个参数为：数据类型，行，列
    // 声明一个2*3的float矩阵
    Eigen::Matrix<float, 2, 3> matrix_23;//float类型
    
    //向量 Eigen::Vector3d 
    // 同时，Eigen 通过 typedef 提供了许多内置类型，不过底层仍是Eigen::Matrix
    // 例如 Vector3d 实质上是 Eigen::Matrix<double, 3, 1>，即三维向量
    Eigen::Vector3d v_3d;//double类型
	// 这是一样的
    Eigen::Matrix<float,3,1> vd_3d;//float类型

    // Matrix3d 实质上是 Eigen::Matrix<double, 3, 3>
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero(); //零矩阵
    // MatrixXd::Identity() 单位矩阵  Eigen::Matrix3d::Random(); 随机数矩阵  MatrixXd::Ones(rows,cols)     
    // 均可以 用C.setXXX 设置  C.setIdentity(rows,cols)   设置单位矩阵
    // 向量初始化  VectorXd::LinSpaced(size,low,high)  // 线性分布
    // 如果不确定矩阵大小，可以使用动态大小的矩阵  建议大矩阵使用 
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > matrix_dynamic;
    // 更简单的
    Eigen::MatrixXd matrix_x;
    // 这种类型还有很多，我们不一一列举

    // 下面是对Eigen阵的操作
    // 输入数据（初始化）
    //  在Eigen中重载了”<<”操作符
    // 通过该操作符即可以一个一个元素的进行赋值，
    // 也可以一块一块的赋值。
    // 另外也可以使用下标进行赋值。
    //matrix_23 << 1, 2, 3, 4, 5, 6;
    matrix_23 << 2,3,4,5,6;  //注意常量矩阵的赋值
    // 正常矩阵形式输出
    cout << matrix_23 << endl;

    // 用()访问矩阵中的元素
    // 针对向量还提供”[]”操作符，注意矩阵则不可如此使用
    for (int i=0; i<2; i++) {
        for (int j=0; j<3; j++)
            cout<<matrix_23(i,j)<<"\t";//每行元素的分隔符
        cout<<endl;//换行
    }

    // 矩阵和向量相乘（实际上仍是矩阵和矩阵）
    v_3d << 3, 2, 1;//double 类型
    vd_3d << 4,5,6;//float 类型
    // 但是在Eigen里你不能混合两种不同类型的矩阵，像这样是错的
    // Eigen::Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;
    // 应该显式转换 matrix_23.cast<double>   float类型转换成 double类型
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << result << endl;
   // float类型 * float　类型
    Eigen::Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
    cout << result2 << endl;

    // 同样你不能搞错矩阵的维度
    // 试着取消下面的注释，看看Eigen会报什么错
    // Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;

    // 一些矩阵运算
    // 四则运算就不演示了，直接用+-*/即可。
    matrix_33 = Eigen::Matrix3d::Random();      // 随机数矩阵
    cout << "Random :Matrix3d matrix_33 =\n" << matrix_33 << endl << endl;

    cout << "matrix_33.transpose =\n" << matrix_33.transpose() << endl;      // 转置
    cout << "matrix_33.sum=\n" <<  matrix_33.sum() << endl;            // 各元素和
    cout << "matrix_33.trace=\n" << matrix_33.trace() << endl;          // 迹
    cout << 10*matrix_33 << endl;               // 数乘
    cout << matrix_33.inverse() << endl;        // 逆
    cout << matrix_33.determinant() << endl;    // 行列式

    // 特征值
    // 实对称矩阵可以保证对角化成功
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver ( matrix_33.transpose()*matrix_33 );
    cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;//特征值
    cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;//特征向量

    // 解方程
    // 我们求解 matrix_NN * x = v_Nd 这个方程
    // N的大小在前边的宏里定义，它由随机数生成
    // 直接求逆自然是最直接的，但是求逆运算量大

    Eigen::Matrix< double, MATRIX_SIZE, MATRIX_SIZE > matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random( MATRIX_SIZE, MATRIX_SIZE );//随机变量初始化
    Eigen::Matrix< double, MATRIX_SIZE,  1> v_Nd;        //列向量
    v_Nd = Eigen::MatrixXd::Random( MATRIX_SIZE,1 ); //随机变量初始化

    clock_t time_stt = clock(); // 计时
    // 直接求逆
    Eigen::Matrix<double,MATRIX_SIZE,1> x = matrix_NN.inverse()*v_Nd;
    //cout << "x = \n" << x << endl;
    cout <<"time use in normal inverse is " << 1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms"<< endl;
    
	// 通常用矩阵分解来求，例如QR分解，速度会快很多
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    //cout << "x = \n" << x << endl;
    cout <<"time use in Qr decomposition is " <<1000*  (clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;

    //矩阵分块
    Eigen::Matrix<double,5,5> Matrix_55;
    Matrix_55 = Eigen::MatrixXd::Random(5,5);
    cout<<"Random Matrix_55 :\n"<<Matrix_55<<endl;
    Eigen::Matrix3d matrixI33 = Eigen::Matrix3d::Identity();
    cout<<"Eye matrixI33 :\n"<<matrixI33<<endl;
    Matrix_55.topLeftCorner(3,3)=matrixI33;
    cout<<"Random Matrix_55 topLeft block replace by Eye matrixI33 :\n"<<Matrix_55<<endl;
    
    
    return 0;
}
