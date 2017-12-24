#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>//莱文贝格－马夸特方法（Levenberg–Marquardt algorithm）能提供数非线性最小化（局部最小）的数值解。
#include <g2o/core/optimization_algorithm_gauss_newton.h>//高斯牛顿法
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>//矩阵库
#include <opencv2/core/core.hpp>//opencv2
#include <cmath>//数学库
#include <chrono>//时间库

// 图优化   http://www.cnblogs.com/gaoxiang12/p/5244828.html
// http://blog.csdn.net/u012525173/article/details/70332103
// http://blog.csdn.net/heyijia0327/article/details/47813405
//数值优化算法
/*
 * 原理介绍
 http://blog.csdn.net/liu14lang/article/details/53991897
 * eigen3例子
 * http://blog.csdn.net/caimagic/article/details/51397285
 
###############
 牛顿法：
 求 f(x)=0
 f(x+det) = f(x)  + f'(x)*det=0  一阶泰勒展开
 det = x - x0 = -f(x0)/f'(x0)    
 迭代公式：
 
 xn  =  xn-1 -  f(xn-1)/f'(xn-1)          通过迭代将零点邻域内一个任选的点收敛至该零点 
#################### 
 牛顿下山法：
 xn  =  xn-1 -  w * f(xn-1)/f'(xn-1)    w =1,. 逐次减半..，0    调整步长逐步降低  避免跳过最优接
 
#####################
 最优化问题
 min(f(x))     求 f'(x)=0
 xn =  xn-1 -  f'(xn-1)/f''(xn-1)      迭代公式   
 
######################
高斯牛顿法
在讲牛顿法的时候，我们举的例子x是一维的，若如果我们遇到多维的x该如何办呢？
这时我们就可以利用雅克比，海赛矩阵之类的来表示高维求导函数了。
比如
f(X)=0,其中X=[x0,x1,...,xn]
所以我们有雅克比矩阵 Jf：  n*n   第一列 f对x0偏导   第n列 f对xn求偏导      一阶偏导数
有海赛矩阵 Hf：                       n*n  对应行列位置ij   f对xi 和 xj 的偏导             二阶偏导

所以高维牛顿法解最优化问题又可写成：
Xn+1=Xn −  (Hf(xn))逆 * Jf * f(xn)

梯度 ∇ Jf   代替了低维情况中的一阶导   
Hessian矩阵代替了二阶导
求逆代替了除法
xn  =  xn-1 -  1/f'(xn-1) * f(xn-1)
而近似下有  Hf(xn)  = Jf 转置 * Jf 
#######################
xn  =  xn-1 -   ( Jf 转置 * Jf )逆 * Jf * f(xn-1)

###########################
Levenberg-Marquardt算法
莱文贝格－马夸特方法（Levenberg–Marquardt algorithm）能提供数非线性最小化（局部最小）的数值解。
此算法能借由执行时修改参数达到结合高斯-牛顿算法以及梯度下降法的优点，
并对两者之不足作改善（比如高斯-牛顿算法之反矩阵不存在或是初始值离局部极小值太远）

在我看来，就是在高斯牛顿基础上修改了一点。
在高斯牛顿迭代法中，我们已经知道
xn  =  xn-1 -   ( Jf 转置 * Jf )逆 * Jf * f(xn-1)

在莱文贝格－马夸特方法算法中则是
xn  =  xn-1 -   ( Jf 转置 * Jf + lamd * 单位矩阵  )逆 * Jf * f(xn-1)

然后Levenberg-Marquardt方法的好处就是在于可以调节:
如果下降太快，使用较小的λ，使之更接近高斯牛顿法
如果下降太慢，使用较大的λ，使之更接近梯度下降法


 */

using namespace std; 
/*
 g2o全称general graph optimization，是一个用来优化非线性误差函数的c++框架。
 */

// 待优化变量——曲线模型的顶点，模板参数：优化变量维度　和　数据类型
class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl() // 重置
    {
        _estimate << 0,0,0;
    }
    
    virtual void oplusImpl( const double* update ) // 更新
    {
        _estimate += Eigen::Vector3d(update);
    }
    //虚函数  存盘和读盘：留空
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
};

// 误差模型—— 曲线模型的边, 模板参数：观测值维度，类型，连接顶点类型(创建的顶点)
class CurveFittingEdge: public g2o::BaseUnaryEdge<1,double,CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge( double x ): BaseUnaryEdge(), _x(x) {}
    // 计算曲线模型误差
    void computeError()
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0,0) = _measurement - std::exp( abc(0,0)*_x*_x + abc(1,0)*_x + abc(2,0) ) ;
    }
    // 存盘和读盘：留空
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
public:
    double _x;  // x 值， y 值为 _measurement
};

int main( int argc, char** argv )
{
    double a=1.0, b=2.0, c=1.0;         // 真实参数值
    int N=100;                                     // 数据点
    double w_sigma=1.0;                  // 噪声Sigma值
    cv::RNG rng;                                 // OpenCV随机数产生器
    double abc[3] = {0,0,0};              // abc参数的估计值

    vector<double> x_data, y_data;      // 数据
    
    cout<<"generating data: "<<endl;
    for ( int i=0; i<N; i++ )
    {
        double x = i/100.0;
        x_data.push_back ( x );
        y_data.push_back (exp ( a*x*x + b*x + c ) + rng.gaussian ( w_sigma ));//加上高斯噪声
        cout<<x_data[i]<<"\t"<<y_data[i]<<endl;
    }
    
    // 构建图优化解决方案，先设定g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<3,1> > Block;  // 每个误差项优化变量维度为3，误差值维度为1
    // 线性方程求解器
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); // 线性方程求解器
    // 矩阵块求解器
    Block* solver_ptr = new Block( linearSolver );      // 矩阵块求解器
    // 梯度下降方法，从GN, LM, DogLeg 中选
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
    // g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm( solver );   // 设置求解器
    optimizer.setVerbose( true );       // 打开调试输出
    
    // 往图中增加顶点
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate( Eigen::Vector3d(0,0,0) );
    v->setId(0);
    optimizer.addVertex( v );
    
    // 往图中增加边
    for ( int i=0; i<N; i++ )
    {
        CurveFittingEdge* edge = new CurveFittingEdge( x_data[i] );
        edge->setId(i);
        edge->setVertex( 0, v );                // 设置连接的顶点
        edge->setMeasurement( y_data[i] );      // 观测数值
        edge->setInformation( Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma) ); // 信息矩阵：协方差矩阵之逆
        optimizer.addEdge( edge );
    }
    
    // 执行优化
    cout<<"start optimization"<<endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;
    
    // 输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout<<"estimated model: "<<abc_estimate.transpose()<<endl;
    
    return 0;
}
