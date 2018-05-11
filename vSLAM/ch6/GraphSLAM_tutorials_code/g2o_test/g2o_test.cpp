#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"

#include "g2o/types/slam3d/vertex_se3.h"
//#include "g2o/types/slam3d/edge_se3.h"
// 使用 宏函数 声明边和顶点类型，注意注释掉了上面两个头文件 
G2O_USE_TYPE_GROUP(slam3d);
//G2O_USE_TYPE_GROUP(slam2d); //2d平面

#include <iostream>

using namespace std;
using namespace g2o;

#define MAXITERATION 50
int main()
{
    cout<< "Hello g2o"<<endl;
    // create the linear solver
    BlockSolverX::LinearSolverType * linearSolver = new LinearSolverCSparse<BlockSolverX::PoseMatrixType>();

    // create the block solver on the top of the linear solver
    BlockSolverX* blockSolver = new BlockSolverX(linearSolver);
    
    //create the algorithm to carry out the optimization
    OptimizationAlgorithmLevenberg* optimizationAlgorithm = new OptimizationAlgorithmLevenberg(blockSolver);

/*  //如果没用前面的宏函数，而是调用的是edge_se3和vertex_se3头文件
    //想让程序能识别VertexSE3这些数据类型，就要显示的调用它们，如下
    //如果只用了头文件，而没用显示调用，那么这些数据类型将不会link进来
    //在下面的optimizer.load函数将不能识别这些数据类型
    for(int f=0; f<10;++f)
    {
        VertexSE3* v = new VertexSE3;
        v->setId(f++);
    }
*/
    // create the optimizer
    SparseOptimizer optimizer;
    
    if(!optimizer.load("../data/sphere_bignoise_vertex3.g2o"))
    {
        cout<<"Error loading graph"<<endl;
        return -1;
    }else
    {
        cout<<"Loaded "<<optimizer.vertices().size()<<" vertices"<<endl;
        cout<<"Loaded "<<optimizer.edges().size()<<" edges"<<endl;
    }

    //优化过程中，第一个点固定，不做优化; 也可以不固定。
    VertexSE3* firstRobotPose = dynamic_cast<VertexSE3*>(optimizer.vertex(0));
    firstRobotPose->setFixed(true);

    optimizer.setAlgorithm(optimizationAlgorithm);
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    cerr<<"Optimizing ..."<<endl;
    optimizer.optimize(MAXITERATION);
    cerr<<"done."<<endl;

    optimizer.save("../data/sphere_after.g2o");
    //optimizer.clear();

    return 0;
}
