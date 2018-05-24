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
* 跟踪的第一步就是 为当前帧 旋转 跟踪的参考帧
### 2.1 包含 每一层金字塔上的：
      * 1.   关键点位置坐标                  posData[i]          (x,y,z)      Eigen::Vector3f*
      * 2.   关键点像素梯度信息              gradData[i]         (dx, dy)     Eigen::Vector2f*
      * 3.   关键点像素值 和 逆深度方差信息   colorAndVarData[i]  (I, Var)     Eigen::Vector2f*
      * 4.   关键点位置对应的 灰度像素点      pointPosInXYGrid[i]  x + y*width  int*
      *       上面四个都是指针数组
      * 5.   产生的 物理世界中的点的数量      numData[i]                        int
### 2.2 产生
      * 1. 帧坐标系下的关键点　3d 位置坐标 的产生  由深度值和像素坐标
      *       P*T*K =  (u,v,1)    P 世界坐标系下的点坐标
      *       Q*I*K =  (u,v,1)    Q 当前坐标系下的点坐标
      *       Q  =  (X/Z，Y/Z，1)  这里Z就相当于 当前相机坐标系下 点的Z轴方向的深度值D
      * 
      *        (X，Y，Z) = D  *  (u,v,1)*K逆 =1/(1/D)* (u*fx_inv + cx_inv, v+fy_inv+cy_inv, 1)
      * 
      *       *posDataPT = (1.0f / pyrIdepthSource[idx]) * Eigen::Vector3f(fxInvLevel*x+cxInvLevel,fyInvLevel*y+cyInvLevel,1);
      * 
      * 2. 关键点像素梯度信息   的产生 
      *     在帧类中有计算直接取来就行了 
      *     *gradDataPT = pyrGradSource[idx].head<2>();
      * 
      * 3.  关键点像素值 和 逆深度方差信息
      *     分别在 帧的 图像金字塔 和 逆深度方差金字塔中已经存在，直接取过来
      *     *colorAndVarDataPT = Eigen::Vector2f(pyrColorSource[idx], pyrIdepthVarSource[idx]);
      * 4. 关键点位置对应的 灰度像素点   直接就是像素所在的位置编号  x + y*width
      * 
      * 5. 产生的 物理世界中的点的数量
      *    首尾指针位置之差就是  三维点 的数量
      * 	numData[level] = posDataPT - posData[level]; 



## 3. 欧式变换矩阵 R,t    跟踪求解   SE3Tracker.cpp
       这里　是　SE3跟踪求解　　欧式变换矩阵求解　

      * LSD-SLAM在位姿估计中采用了直接法，也就是通过最小化光度误差来求解两帧图像间的位姿变化。
      * 并且采用了LM算法迭代求解。
      * LSD-SLAM在求解两帧图像间的SE3变换主要在SE3Tracker类中进行实现。
### 该类中有四个函数比较重要，分别为
      *  1.  SE3Tracker::trackFrame                          主调函数
      *  2. SE3Tracker::calcResidualAndBuffers      被调用计算　参考帧3d点变换到当前帧图像坐标系下的残差(灰度误差)　和　梯度(对应点像素梯度)　
      *  3. SE3Tracker::calcWeightsAndResidual　 被调用计算
      *  4. SE3Tracker::calculateWarpUpdate

### 1.  SE3Tracker::trackFrame  
      *  图像金字塔迭代level-4到level-1
      *	Step1: 对参考帧当前层构造点云(reference->makePointCloud) 
      *                  利用逆深度信息　和　相应的像素坐标 以及相机内参数得到　在参考帧坐标下的3d点    
      *                  (X，Y，Z) = D  *  (u,v,1)*K逆 =1/(1/D)* (u*fx_inv + cx_inv, v+fy_inv+cy_inv, 1)
      *	Step2: 计算变换到当前帧的残差和梯度(calcResidualAndBuffers)
      *                 计算参考帧3d点变换到当前帧图像坐标系下的残差(灰度误差)　和　梯度(对应点像素梯度)　
      *                  1. 参考帧3d点 R，t变换到 当前相机坐标系下 
      *                         Eigen::Vector3f Wxp = rotMat * (*refPoint) + transVec;
      * 		    2. 再 投影到 当前相机 像素平面下
      * 		            float u_new = (Wxp[0]/Wxp[2])*fx_l + cx_l;// float 浮点数类型
      * 		            float v_new = (Wxp[1]/Wxp[2])*fy_l + cy_l;
      *                  3. 根据浮点数坐标 四周的四个整数点坐标的梯度 使用位置差加权进行求和得到亚像素灰度值　和　亚像素　梯度值
      *                      	int ix = (int)x;// 取整数    左上方点
      *				int iy = (int)y;
      *				float dx = x - ix;// 差值
      *				float dy = y - iy;
      *				float dxdy = dx*dy;
      *				// map 像素梯度  指针
      *				const Eigen::Vector4f* bp = mat +ix+iy*width;// 左上方点　像素位置 梯度的 指针
      *				// 使用 左上方点 右上方点  右下方点 左下方点 的灰度梯度信息 以及 相应位置差值作为权重值 
                                 线性加权获取　亚像素梯度信息
      *				return dxdy * *(const Eigen::Vector3f*)(bp+1+width)// 右下方点
      *					+ (dy-dxdy) * *(const Eigen::Vector3f*)(bp+width)// 左下方点
      *					+ (dx-dxdy) * *(const Eigen::Vector3f*)(bp+1)// 右上方点
      *						+ (1-dx-dy+dxdy) * *(const Eigen::Vector3f*)(bp);// 左上方点
      * 　　　　  4.  记录变换　
      * 　　　　　　　　a. 对参考帧灰度　进行一次仿射变换后　和　当前帧亚像素灰度做差得到　残差(灰度误差)
      *                                b. 对两帧下　3d点坐标的　Z轴深度值　的变化比例
      *	Step3: 计算 方差归一化加权残差和　使用误差方差对方差归一化 后求和 (calcWeightsAndResidual)
      * 
      *	Step4: 计算雅克比向量以及A和b(calculateWarpUpdate)
      * 
      *	Step5: 计算得到收敛的delta，并且更新SE3(inc = A.ldlt().solve(b))
      * 
      *	重复Step2-Step5直到收敛或者达到最大迭代次数
      * 
      * 计算下一层金字塔

 
## 4. 相似变换矩阵 sR, t  跟踪求解   Sim3Tracker.cpp

## 5. 跟丢之后的重定位               Relocalizer.cpp
