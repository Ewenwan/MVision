/**
* This file is part of LSD-SLAM.
* 最小二乘 优化  更新 update
*  参考 https://blog.csdn.net/lancelot_vim/article/details/51758870
*  第一个类型LSGX，这里面有3个类，分别是LSG4,LSG6和LSG7，
* 他们定义了4个参数6个参数以及7个参数的最小二乘法  优化算法
* 
* X86框架下 	SSE 单指令多数据流式扩展 优化
*                     使用SSE内嵌原语  + 使用SSE汇编
* ARM平台 NEON 指令  优化
* 参考 http://zyddora.github.io/2016/02/28/neon_1/
* NEON就是一种基于SIMD思想的ARM技术
* Single Instruction Multiple Data (SIMD)顾名思义就是“一条指令处理多个数据 ， 并行处理技术
* 
* 16×    128-bit 寄存器(Q0-Q15)；
* 或32× 64-bit  寄存器(D0-D31)
* 或上述寄存器的组合。
*  实际上D寄存器和Q寄存器是重叠的
* 
* 所有的支持NEON指令都有一个助记符V，下面以32位指令为例，说明指令的一般格式：
* V{<mod>}<op>{<shape>}{<cond>}{.<dt>}{<dest>}, src1, src2
* 1. <mod>  模式  Q  H   D   R 系列
* 2. <op>     操作   ADD, SUB, MUL    加法  减法  乘法
* 3. <shape> 数据形状  
*                    Long (L),         操作双字vectors，生成四倍长字vectors   结果的宽度一般比操作数加倍，同类型
*                    Wide (W),       操作双字 + 四倍长字，生成四倍长字  结果和第一个操作数都是第二个操作数的两倍宽度
*                    Narrow (N)    操作四倍长字，生成双字  结果宽度一般是操作数的一半
* 5. .<dt>      数据类型        s8, u8, f32 
* 
* 6. <dest>  目的地址 
* 7. <src1>  源操作数1
* 8. <src2>  源操作数2
* 
*/

#pragma once

#include "util/EigenCoreInclude.h"
#include <opencv2/core/core.hpp>



namespace lsd_slam
{


typedef Eigen::Matrix<float, 6, 1> Vector6;// R, t  三个旋转量  三个平移量
typedef Eigen::Matrix<float, 6, 6> Matrix6x6;// 上面对应得协方差矩阵?   雅可比矩阵?

typedef Eigen::Matrix<float, 7, 1> Vector7;// sR, t 一个相似变换比例s   三个旋转量  三个平移量
typedef Eigen::Matrix<float, 7, 7> Matrix7x7;

typedef Eigen::Matrix<float, 4, 1> Vector4;// 
typedef Eigen::Matrix<float, 4, 4> Matrix4x4;


/**
 * A 6x6 self adjoint matrix with optimized "rankUpdate(u, scale)" (10x faster than Eigen impl, 1.8x faster than MathSse::addOuterProduct(...)).
 *  6维 变量优化更新
 */
class OptimizedSelfAdjointMatrix6x6f
{
public:
  OptimizedSelfAdjointMatrix6x6f();
 // 更新
  void rankUpdate(const Eigen::Matrix<float, 6, 1>& u, const float alpha);
  // 重载加法运算符
  void operator +=(const OptimizedSelfAdjointMatrix6x6f& other);

  void setZero();

  void toEigen(Eigen::Matrix<float, 6, 6>& m) const;
private:
  enum {
    Size = 24
  };
  EIGEN_ALIGN16 float data[Size];
};
/**
 * Builds normal equations and solves them with Cholesky decomposition.
 * 建立正规方程组，并用Cholesky分解法求解。
 */
class NormalEquationsLeastSquares
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  OptimizedSelfAdjointMatrix6x6f A_opt;
  Matrix6x6 A;
  Vector6 b;

  bool solved;
  float error;
  size_t maxnum_constraints, num_constraints;

  virtual ~NormalEquationsLeastSquares();

  virtual void initialize(const size_t maxnum_constraints);
  // 矩阵更新 
  virtual void update(const Vector6& J, const float& res, const float& weight = 1.0f);
  virtual void finish();
  virtual void finishNoDivide();
  virtual void solve(Vector6& x);

  void combine(const NormalEquationsLeastSquares& other);
};



/**
 * Builds 4dof LGS (used for depth-lgs, at it has only 7 non-zero entries in jacobian)
 * only used to accumulate data, NOT really as LGS
 * 4维 变量优化更新
 */
class NormalEquationsLeastSquares4
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Matrix4x4 A;
  Vector4 b;

  bool solved;
  float error;
  size_t maxnum_constraints, num_constraints;

  virtual ~NormalEquationsLeastSquares4();

  virtual void initialize(const size_t maxnum_constraints);
  virtual void update(const Vector4& J, const float& res, const float& weight = 1.0f);

  void combine(const NormalEquationsLeastSquares4& other);

  virtual void finishNoDivide();
};


// 7维 变量优化更新
class NormalEquationsLeastSquares7
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Matrix7x7 A;
  Vector7 b;

  bool solved;
  float error;
  size_t maxnum_constraints, num_constraints;

  virtual ~NormalEquationsLeastSquares7();

  virtual void initializeFrom(const NormalEquationsLeastSquares& ls6, const NormalEquationsLeastSquares4& ls4);
  void combine(const NormalEquationsLeastSquares7& other);
};

}
