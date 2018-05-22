# 跟踪参考帧 线程

## 1. 最小二乘优化求解 least_squares.cpp
      4个参数6个参数以及7个参数的最小二乘法  优化算法

      A.noalias() += J * J.transpose() * weight;// eigen 的 noalias()机制 避免中间结果 类的 构造
      b.noalias() -= J * (res * weight);
      error += res * res * weight;
 
### 使用了一些 指令集优化算法 来提高 矩阵运算的 速度
#### a. X86框架下 	SSE 单指令多数据流式扩展 优化
      *  使用SSE内嵌原语  + 使用SSE汇编
      * 
      * 
      
      
#### b. ARM平台 NEON 指令  优化
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
      
## 2. 参考帧的选择 跟踪参考帧        TrackingReference.cpp 


## 3. 欧式变换矩阵 R,t    跟踪求解   SE3Tracker.cpp

 
## 4. 相似变换矩阵 sR, t  跟踪求解   Sim3Tracker.cpp

## 5. 跟丢之后的重定位               Relocalizer.cpp
