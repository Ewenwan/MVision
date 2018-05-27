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
      * 误差函数　E(se3) = SUM((Iref(pi) - I(W(pi,Dref(pi),se3))^2)
      * (参考帧下的像素值-参考帧3d点变换到当前帧下的像素值)^2　
      * 也就是 所有匹配点　的光度误差和
      * LSD-SLAM在求解两帧图像间的SE3变换主要在SE3Tracker类中进行实现。
### 求解步骤
      * 1. 首先在参考帧下根据深度信息计算出3d点云
      * 2. 其次，计算参考帧下的点变换到当前帧下的亚像素像素值，进而计算初始　光度匹配误差err
      * 3. 再次，利用误差对单个3d点的导数信息计算单个误差的可信程度，进行加权后方差归一化，
      *　　得到权重ｗ，为了减少外点（outliers）对算法的影响
      * 4. 然后，利用误差函数对误差向量求导数，求得各个点的误差之间的协方差矩阵，
           也是雅克比矩阵Ｊ,计算系数A 和 b 
      * 5. 最后计算　变换矩阵　se3的　更新量 dse3　
      * 　　　dse3 = -(J转置*J)逆矩阵*J转置*w*err  直接求逆矩阵计算量较大
      *      也可以写成：
      * 　　　   J转置*J * dse3 = -J转置*w*error
      *       紧凑形式：
      * 　　　   A * dse3 = b
      *       使用　LDLT分解 A = LDU＝LDL转置＝LDLT，求解线性方程组　A * x = b
      *       记录　ｕ　＝　LD，　D为对角矩阵　L为下三角矩阵　L转置为上三角矩阵
      *             v  ＝  L转置
      *       将A分解为上面两个矩阵　相乘　A = u * v
      *       Ax = b就可以化为u(v*x) = u*y = b
      *       先求解　y
      *       得到　  y = v*x
      *       再求解　x 即　dse3 是欧式变换矩阵se3的更新量
      * 
      * 6. 然后对　变换矩阵　左乘　se3指数映射　进行　更新　
      * 　　se3　指数映射到SE3 通过　李群乘法直接　对　变换矩阵　左乘更新　

      
### 该类中有四个函数比较重要，分别为
      *  1. SE3Tracker::trackFrame              主调函数
      *  2. SE3Tracker::calcResidualAndBuffers  被调用计算　
      *     初始误差　err 参考帧3d点变换到当前帧图像坐标系下的残差(灰度误差)　和　梯度(对应点像素梯度)　
      *  3. SE3Tracker::calcWeightsAndResidual　被调用计算    
      *     误差可信度权重w　并对误差方差归一化得到加权平均后的误差
      *  4. SE3Tracker::calculateWarpUpdate　　 被调用计算    
      *　　 误差关系矩阵　协方差矩阵　雅克比矩阵J　以及A=J转置*J b=-J转置*w*error 求接线性方程组

###  SE3Tracker::trackFrame  
      *  图像金字塔迭代level-4到level-1
      
      *       Step1: 对参考帧当前层构造点云(reference->makePointCloud) 
      *                  利用逆深度信息　和　相应的像素坐标 以及相机内参数得到　在参考帧坐标下的3d点    
      *                  (X，Y，Z) = D  *  (u,v,1)*K逆 =1/(1/D)* (u*fx_inv + cx_inv, v+fy_inv+cy_inv, 1)
      
      *       Step2: 计算变换到当前帧的残差和梯度(calcResidualAndBuffers)
      *             计算参考帧3d点变换到当前帧图像坐标系下的残差(灰度误差)　和　梯度(对应点像素梯度)　
      *             1. 参考帧3d点 R，t变换到 当前相机坐标系下 
      *           　　　Eigen::Matrix3f rotMat = referenceToFrame.rotationMatrix();// 旋转矩阵
      *         　　　　Eigen::Vector3f transVec = referenceToFrame.translation();// 平移向量
      *                Eigen::Vector3f Wxp = rotMat * (*refPoint) + transVec;// 欧式变换
      *             2. 再 投影到 当前相机 像素平面下
      *                float u_new = (Wxp[0]/Wxp[2])*fx_l + cx_l;// float 浮点数类型
      *                float v_new = (Wxp[1]/Wxp[2])*fy_l + cy_l;
      *             3. 根据浮点数坐标 四周的四个整数点坐标的梯度 使用位置差加权进行求和得到亚像素灰度值　和　亚像素　梯度值
      *                 x=u_new;
      *                 y=v_new;
      *                 int ix = (int)x;// 取整数    左上方点
      *                 int iy = (int)y;
      *                 float dx = x - ix;// 差值
      *                 float dy = y - iy;
      *                 float dxdy = dx*dy;
      *                 // map 像素梯度  指针
      *                 const Eigen::Vector4f* bp = mat +ix+iy*width;// 左上方点　像素位置 梯度的 指针
      *                 // 使用 左上方点 右上方点  右下方点 左下方点 的灰度梯度信息 
      *                 // 以及 相应位置差值作为权重值 线性加权获取　亚像素梯度信息
      *                 resInterp=dxdy * *(const Eigen::Vector3f*)(bp+1+width)// 右下方点
      *                          + (dy-dxdy) * *(const Eigen::Vector3f*)(bp+width)// 左下方点
      *                          + (dx-dxdy) * *(const Eigen::Vector3f*)(bp+1)// 右上方点
      *                          + (1-dx-dy+dxdy) * *(const Eigen::Vector3f*)(bp);// 左上方点
      * 
      *               需要注意的是，在给梯度变量赋值的时候，这里乘了焦距
      * 　　　　       这里求得　亚像素梯度之后　又乘上了　相机内参数，因为在求　误差偏导数时需要　需要用到　dx*fx　　dy*fy  
      *               *(buf_warped_dx+idx) = fx_l * resInterp[0];// 当前帧匹配点亚像素 梯度　gx = dx*fx  
      *               *(buf_warped_dy+idx) = fy_l * resInterp[1];// 匹配点亚像素 梯度               gy = dy*fy
      *               *(buf_warped_residual+idx) = residual;// 对应 匹配点 像素误差
      *               这里的　  dx= resInterp[0],  dy =  resInterp[0] 亚像素梯度
      * 
      *                (buf_d+idx) = 1.0f / (*refPoint)[2];  // 参考帧 Z轴 倒数  = 参考帧逆深度
      *               *(buf_idepthVar+idx) = (*refColVar)[1];// 参考帧逆深度方差
      * 　　　　    4.  记录变换　
      * 　　　　　　　a. 对参考帧灰度　进行一次仿射变换后　和　当前帧亚像素灰度做差得到　残差(灰度误差)
      *                   现在求得的残差只是纯粹的光度误差
      * 　　　　　　　　   float c1 = affineEstimation_a * (*refColVar)[0] + affineEstimation_b;// 参考帧 灰度[0]  仿射变换后
      *                   float c2 = resInterp[2];//  当前帧 亚像素 第三维度是图像 灰度值
      *                   float residual = c1 - c2;// 匹配点对 的灰度  误差值　　
      *                                      
      *                 疑问1：代码中对参考帧的灰度做了一个仿射变换的处理，这里的原理是什么？
      *                     在SVO代码中的确有考虑到两帧见位姿相差过大，因此通过在空间上的仿射变换之后再求灰度的操作。
      *                     但是在这里的代码中没有看出具体原理。
      *              b. 对两帧下　3d点坐标的　Z轴深度值　的变化比例
      *                     float depthChange = (*refPoint)[2] / Wxp[2];
      *                     usageCount += depthChange < 1 ? depthChange : 1;//   记录深度改变的比例  
      *　　　　　　   c. 匹配点总数
      *                  buf_warped_size = idx;// 匹配点数量＝　包括好的匹配点　和　不好的匹配点　数量
      
      *         Step3: 计算 方差归一化加权残差和　使用误差方差　对　误差归一化 后求和 (calcWeightsAndResidual)
      *              计算误差权重，归一化方差以及Huber-weight，最终把这个系数存在数组　buf_weight_p　中
      *　　　　　　   每个像素点都有一个光度匹配误差，而每个点匹配的可信度根据其偏导数可以求得，
      *　　　　　　   加权每一个误差　然后求所有点　光度匹配误差的均值
      * 　　　　      误差导数　drpdd = dr= -(dx*fx　*　(tx*z' - tz*x')/(z'^2*d)  + dy*fy　*　(ty*z' - tz*y')/(z'^2*d))
      *                                = -(gx　*　(tx*z' - tz*x')/(z'^2*d)  + gy　*　(ty*z' - tz*y')/(z'^2*d))
      *                      dx, dy 为参考帧3d点映射到当前帧图像平面后的梯度
      *             　　　　　fx,  fy 相机内参数
      *             　　　　　tx,ty,tz 为参考帧到当前帧的平移变换(忽略旋转变换)　
                                      t  = translation()
      *            　　　　　 x',y',z'  为参考帧3d点变换到当前帧坐标系下的3d点     
                                       x' = Wxp[0] , y' = Wxp[1] , z' = Wxp[2] 
      *             　　　　　d = d = *(buf_d) 为参考帧3d点在参考帧下的逆深度
      *                      gx = *(buf_warped_dx)
      *                      gy = *(buf_warped_dy)
      *              方差平方   float s = settings.var_weight  *  *(buf_idepthVar+i);//    参考帧 逆深度方差  平方
      * 　           误差权重为加权方差平方的倒数  float w_p = 1.0f / ((cameraPixelNoise2) + s * drpdd * drpdd);// 误差权重 方差倒数
      *              初步误差　　　　             rp =*( buf_warped_residual) ; //初步误差
      *              加权的误差                  float weighted_rp = fabs(rp*sqrtf(w_p));// |r|加权的误差
      *              Huber-weight权重  
      *              float wh = fabs(weighted_rp < (settings.huber_d/2) ? 1 : (settings.huber_d/2) / weighted_rp);
      *              记录　*(buf_weight_p+i) = wh * w_p;    // 乘上 Huber-weight 权重 得到最终误差权重避免外点的影响
      * 　　　       求和　 sumRes += wh * w_p * rp*rp;　// 加权误差和
      *              取均值　return sumRes / buf_warped_size;// 返回加权误差均值　　　　　　　　　
                     lastErr =  sumRes / buf_warped_size;
                  
      *      Step4: 计算雅克比向量以及A和b(calculateWarpUpdate)
      * 　　　　　　 J转置*J * dse3 = -J转置*w*lastErr
      *  　　　　　  对于参考帧中的一个3D点pi位置处的残差求雅可比，这里需要使用链式求导法则 
      *                       Ji =      1/z'*dx*fx + 0* dy *fy
      *                                  0* dx *fx +  1/z' *dy *fy
      *                         -1 *    -x'/z'^2 *dx*fx  - y'/z'^2 *dy*fy 
      *                                 -x'*y'/z'^2 *dx*fx - (1+y'^2/z'^2) *dy*fy
      *                                 (1+x'^2/z'^2)*dx*fx + x'*y'/z'^2*dy*fy
      *                                - y'/z' *dx*fx + x'/z' * dy*fy
      *
      *                       A = J *J转置*(*(buf_weight_p+i))
      *                       b = -J转置*(*(buf_weight_p+i))*lastErr
      * 
      *      Step5: 计算得到收敛的delta，并且更新SE3(dse3 = A.ldlt().solve(b))
      * 　　　　 for(int i=0;i<6;i++) A(i,i) *= 1+LM_lambda;// (A+λI)  　列文伯格马尔夸克更新算法　LM 调整量　
      *          Vector6 inc = A.ldlt().solve(b);// LDLT矩阵分解求解　dse3 李代数更新量
                 Sophus::SE3f new_referenceToFrame = Sophus::SE3f::exp((inc)) * referenceToFrame;　
      * 　　　　
      *      重复Step2-Step5直到收敛或者达到最大迭代次数
      * 
      *  计算下一层金字塔


 
## 4. 相似变换矩阵 sR, t  跟踪求解   Sim3Tracker.cpp

## 5. 跟丢之后的重定位               Relocalizer.cpp
