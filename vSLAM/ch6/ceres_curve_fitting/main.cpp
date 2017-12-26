#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>//最小二成非线性优化
#include <chrono>//计时
#include <fstream>//文件流
/*
 * https://www.cnblogs.com/shang-slam/p/6821560.html
 * http://blog.csdn.net/liminlu0314/article/details/16808239
 * http://ceres-solver.org/features.html
 　Ceres Solver是谷歌2010就开始用于解决优化问题的C++库，2014年开源．
 　在Google地图，Tango项目，以及著名的SLAM系统OKVIS和Cartographer的优化模块中均使用了Ceres Solver.
 　
 　众所周知，最大似然估计（MLE）是许多机器人和计算机视觉应用中使用的估计方法。 
 　在高斯假设下，MLE转换为非线性最小二乘（NLS）问题。
 　存在有效的NLS解决方案，它们是基于迭代求解稀疏线性系统直到收敛。
 　
 　在SLAM领域优化问题还可以使用g2o来求解．不过Ceres提供了自动求导功能，虽然是数值求导，
 　但可以避免复杂的雅克比计算，目前来看Ceres相对于g2o的缺点仅仅是依赖的库多一些（g2o仅依赖Eigen）．
 　但是提供了可以直接对数据进行操作的能力，相对于g2o应用在视觉SLAM中，
 　更加倾向于通用的数值优化，最重要的是提供的官方资料比较全（看g2o简直受罪．．．）．
 　详细的介绍可以参考google的文档：http://ceres-solver.org/features.html
 　
 　
 　优化问题的本质是调整优化变量，使得优化变量建模得出的估计值不断接近观测数据（使得目标函数下降），
 　是最大似然框架下对优化变量的不断调整，得到优化变量建模得出的估计值在观测条件下的无偏估计过程．
 　
 　ObjectiveFunction：目标函数
 　ResidualBlock：残差（代价函数的二范数，有时不加区分），多个ResidualBlock组成完整的目标函数；
 　CostFunction：代价函数，观测数据与估计值的差，观测数据就是传感器获取的数据，估计值是使用别的方法获取
			      （例如初始化，ICP，PnP或者匀速模型．．．）的从优化变量通过建模得出的观测值；例如从对极几何得到的相机位姿，
			      三角化得到的地图点可以作为优化变量的初始值，但是需要利用坐标系变换和相机模型转化为2D平面上的像素坐标估计值，
			      与实际测量得到的观测值之间构建最小二乘问题；
    ParameterBlock：优化变量；
    LossFunction：核函数，用来减小Outlier的影响，对应g2o中的edge->setRobustKernel()		      
 　
 　
 */

using namespace std;

// [1] 代价函数的计算模型
// １．定义一个Functor(拟函数/函数对象)类，其中定义的是CostFunction. 需要重载函数调用运算符，
//  从而可以像使用函数一样使用该类对象．（与普通函数相比，能够在类中存储状态，更加灵活）
struct CURVE_FITTING_COST//曲线拟合代价函数
{
    CURVE_FITTING_COST ( double x, double y ) : _x ( x ), _y ( y ) {}//直接赋值   _x = x;     _y  = y;
    /*
     函数　y=exp(a*x^2 +b*x+c) + w//w为噪声
    参差　 y-exp(a*x^2 +b*x+c)
     */
    // 残差的计算
    template <typename T>//必须使用模板类型   通用参数类型
    bool operator() (            // 必须要编写一个重载() 运算
      //所有的输入参数和输出参数都要使用T类型
        const T* const abc,     // 模型参数，有3维
        T* residual ) const      // 残差
    { // T ( _y )  T ( _x ) 强制类型转换
        residual[0] = T ( _y ) - ceres::exp ( abc[0] * T ( _x )  * T ( _x ) + abc[1] * T ( _x ) + abc[2] ); 
	    // y-exp(a * x^2+b * x + c)
        return true;//必须返回ture
    }
     private:   //自添加
  // 观测值
    const double _x, _y;    //常量 double类型  x,y数据
};

int main ( int argc, char** argv )
{
    double a=1.0, b=2.0, c=1.0;         // 真实参数值
    int N=100;                          // 数据点数量
    double w_sigma=1.0;                 // 噪声Sigma值  高斯分布方差
    cv::RNG rng;                        // OpenCV随机数产生器
    double abc[3] = {0,0,0};            // abc参数的估计值

    vector<double> x_data, y_data;      // 数据 容器

    stringstream ss;//字符串流
    
    cout<<"生成数据 generating data: "<<endl;
    for ( int i=0; i<N; i++ )
    {
        double x = i/100.0;//自变量
        x_data.push_back ( x );
        y_data.push_back (
            exp ( a * x * x + b * x  + c ) + rng.gaussian ( w_sigma )//加上高斯噪声
        );
        cout<<x_data[i]<<"\t"<<y_data[i]<<endl;
	ss << x_data[i] << " " << y_data[i] << endl;//定向到 字符串流
    }
    
    //将生成的点坐标保存到points.txt
    ofstream file("points.txt"); 
    file << ss.str();
    
    // 构建最小二乘问题
    //声明一个残差方程，CostFunction通过模板类AutoDiffCostFunction来进行构造，
    //第一个模板参数为残差对象，也就是最开始写的那个那个带有重载()运算符的结构体，第二个模板参数为残差个数，第三个模板参数为未知数个数，最后参数是结构体对象。
    ceres::Problem problem;
    for ( int i=0; i<N; i++ )
    {
        problem.AddResidualBlock (     // 向问题中添加误差项
        // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
        // AutoDiff 自动求导                         指定　　误差项维度１　　　参数３维　　　　
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3> (  // 模型参数，有3维
                new CURVE_FITTING_COST ( x_data[i],  y_data[i] )//带入误差方差
            ),
            nullptr,            // 核函数，这里不使用，为空
            abc                  // 待估计参数 这里为 数组 输入的为 地址   如果是常量 需要 &x  取地址
        );
    }

    // 配置求解器
    // 这个类有许多字段，每个字段都提供了一些枚举值供用户选择。所以需要时只要查一查文档就知道怎么设置了。
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR;  // QR分解  增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 优化过程信息 输出到 标志输出
  /** 参数选择
    options.gradient_tolerance = 1e-16;
    options.function_tolerance = 1e-16;
    ...
    梯度阈值　gradient_tolerance．
    相邻两次迭代之间目标函数之差　function_tolerance．
    梯度下降策略　trust_region_strategy　可选levenberg_marquardt，dogleg．
    线性增量方程 HΔx=g 求解方法　linear_solver　可选sparse_schur，dense_schur，sparse_normal_cholesky，
		    视觉SLAM中主要采用稀疏Schur Elimination/ Marginalization的方法（也就是消元法），
		    将地图点的增量边缘化，先求出相机位姿的增量，可以极大地较少计算量，避免H矩阵直接求逆
   稀疏线性代数库　sparse_linear_algebra_library　可选suite_sparse，cx_sparse（ceres编译时需额外编译），
				  cx_sparse相对suite_sparse，更精简速度较慢，但是不依赖BLAS和LAPACK．这个通常选择suite_sparse即可．
  稠密线性代数库　 dense_linear_algebra_library　可选eigen，lapack．
  边缘化次序　ParameterBlockOrdering　设置那些优化变量在求解增量方程时优先被边缘化，一般会将较多的地图点先边缘化，
			  不设置ceres会自动决定边缘化次序，这在SLAM里面常用于指定Sliding Window的范围．	
  多线程　这里的设置根据运行平台会有较大不同，对计算速度的影响也是最多的．
		  分为计算雅克比时的线程数num_threads，以及求解线性增量方程时的线程数num_linear_solver_threads．			  
  迭代次数　max_num_iterations，有时迭代多次均不能收敛，可能是初值不理想或者陷入了平坦的区域等等原因，
		      需要设定一个最大迭代次数．			  			  
   */
    ceres::Solver::Summary summary;                // 优化信息
  //优化求解起始时间
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   // 开始优化   
    ceres::Solve ( options, &problem, &summary );
    //优化求解结束时间  
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"最小二成法优化时间 solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // 输出结果
    cout<<summary.BriefReport() <<endl;//简易的报告
    cout<<"estimated a,b,c = ";
    for ( auto a:abc ) cout<<a<<" ";
    cout<<endl;

    return 0;
}

