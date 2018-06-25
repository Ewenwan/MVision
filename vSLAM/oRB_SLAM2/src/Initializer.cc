/* 单目初始化 单应矩阵 本质矩阵 恢复R t 三角变换求 3D点
* This file is part of ORB-SLAM2
* 
* 单目相机初始化
* 用于平面场景的单应性矩阵H(8中运动假设) 和用于非平面场景的基础矩阵F(4种运动假设)，
* 然后通过一个评分规则来选择合适的模型，恢复相机的旋转矩阵R和平移向量t 和 对应的3D点(尺度问题)。
* 
* 
 *【0】2D-2D点对 求变换矩阵前先进行标准化  去 均值后再除以绝对矩
* 单目初始化特征点 归一化 坐标均值为0  一阶绝对矩为1
* mean_x  =  sum( ui) / N   mean_y =  sum(vi)/N
* 绝对矩  mean_x_dev = sum（abs(ui - mean_x)）/ N     mean_y_dev = sum（abs(vi - mean_y)）/ N 
*
* 绝对矩倒数  sX = 1/mean_x_dev     sY = 1/mean_y_dev 
* 
* 标准化后的点坐标
* u =  (ui - mean_x) × sX
* v =  (vi - mean_y) × sY 
* 
* 标准化矩阵   其逆矩阵×标准化点 得到原始坐标 
*      用于 计算变换矩阵后  从原始坐标计算对称的转换误差 来计算变换矩阵得分
* T =    sX   0    -mean_x * sX
*        0    sY   -mean_y * sY
*        0    0         1
* 
* 标准化矩阵  * 点坐标    =   标准化后的的坐标
*       ui         ui × sX - mean_x * sX   = (ui - mean_x) × sX       u
*  T ×  vi    =    vi  × sY - mean_y * sY  = (vi - mean_y) × sY   =   v
*       1               1              				      1
* 
* 点坐标    =    标准化矩阵 逆矩阵 * 标准化后的的坐标
* 
* 
* 
*【1】 2D- 2D 点对 单应矩阵  H   2D点对其次坐标  间的 转换 矩阵  3*3
*  采用归一化的直接线性变换（normalized DLT）
* * p2 = H *p1 关键点  对的 变换关系矩阵H
 * 一个点对 2个约束
 * 4 点法求解  单应矩阵 H 再对 H进行分解
* 
*【2】 对极几何 求解 基本矩阵 F  两组单目相机 2D图像
* (随机采样序列 8点法求解)
*  2D 点对 求 两相机的 旋转和平移矩阵
 * 空间点 P  两相机 像素点对  p1  p2 两相机 归一化平面上的点对 x1 x2 与P点对应
 * 相机内参数 K  两镜头旋转平移矩阵  R t 或者 变换矩阵 T
 *  p1 = KP  (世界坐标系)     p2 = K( RP + t)  = KTP
 *  而 x1 =  K逆* p1  x2 =  K逆* p2  相机坐标系下 归一化平面上的点     x1= (px -cx)/fx    x2= (py -cy)/fy
 * 所以   x1 = P  得到   x2 =  R * x1  + t   
 * 
 *  t 外积  x2  = t 外积 R * x1 +   t 外积 t  =   t 外积 R * x1   ；t 外积 t   =0    sin(cet) =0 垂线段投影 方向垂直两个向量
 *  x2转置 *  t 外积  x2 = x2转置 * t 外积 R  x1   = 0 ；因为  t 外积  x2 得到的向量垂直 t 也垂直 x2
 *   有 x2转置 * t 外积 R  x1   = x2转置 * E * x1 =  0 ； E 为本质矩阵
 * p2转置 * K 转置逆 * t 外积 R * K逆 * p1   = p2转置 * F * p1 =  0 ；
 * F 为基础矩阵
 * 
 * x2转置 * E * x1 =  0    x1 x2  为 由 像素坐标转化的归一化坐标
 * 一个点对一个约束 ，8点法  可以算出 E的各个元素 ，
 * 再 奇异值分解 E 得到 R t
 * 
 * 
 * 
 * 【3】变换矩阵 评分 方法
 * 和卡方分布的对应值比较，由此判定该点是否为内点。累计内点的总得分
 * SM=∑i( ρM( d2cr(xic,xir,M)   +   ρM( d2rc(xic,xir,M ) )
 *  d2cr 为 2D-2D 点对  通过哦转换矩阵的 对成转换误差
 * 
 * ρM 函数为 ρM(d^2)  = 0                  当  d^2 > 阈值(单应矩阵时 为 5.99  基础矩阵时为 3.84)
 *                     得分上限  - d^2     当  d^2 < 阈值
 *                                        得分上限 均为 5.99 
 * 
 *【4】 从两个模型 H F 得分为 Sh   Sf 中选着一个 最优秀的 模型 的方法为
 * 
 * 文中认为，当场景是一个平面、或近似为一个平面、或者视差较小的时候，可以使用单应性矩阵H，
 * 当场景是一个非平面、视差大的场景时，使用基础矩阵F恢复运动
 * RH=SH /(SH+SF)
 * 当RH大于0.45时，选择从单应性变换矩阵还原运动。
 * 不过ORB_SLAM2源代码中使用的是0.4作为阈值
 * 
 * 
 * 【5】单应矩阵求解
 *   1点 变成 2点   p2   =  H21 * p1
      u2         h1  h2  h3        u1
      v2  =      h4  h5  h6    *   v1
      1          h7  h8  h9        1   
      
      u2 = (h1*u1 + h2*v1 + h3) /( h7*u1 + h8*v1 + h9)
      v2 = (h4*u1 + h5*v1 + h6) /( h7*u1 + h8*v1 + h9)
    
      -((h4*u1 + h5*v1 + h6) - ( h7*u1*v2 + h8*v1*v2 + h9*v2))=0  式子为0  左侧加 - 号不变
        h1*u1 + h2*v1 + h3 - ( h7*u1*u2 + h8*v1*u2 + h9*u2)=0
        
        0    0   0  -u1  -v1  -1   u1*v2   v1*v2    v2
        u1 v1  1    0    0    0   -u1*u2  - v1*u2  -u2    ×(h1 h2 h3 h4 h5 h6 h7 h8 h9)转置  = 0
        
        8对点  约束 A 
        A × h = 0 求h   奇异值分解 A 得到 单元矩阵 H
 * 
 * 【6】单应变换 求 变化距离误差
 *  * 1点 变成 2点   p2   =  H12 * p1
 u2        h11  h12  h13        u1
 v2  =     h21  h22  h23    *   v1
 1         h31  h32  h33         1   第三行 
 
 * 2 点 变成 1点  p1   =  H21 * p2
 u1‘        h11inv   h12inv   h13inv         u2
 v1’  =     h21inv   h22inv   h23inv     *   v2
 1          h31inv   h32inv   h33inv          1    第三行 h31inv*u2+h32inv*v2+h33inv 
 前两行 同除以 第三行 消去非零因子
  p2 由单应转换到 p1
  u1‘ = (h11inv*u2+h12inv*v2+h13inv)* 第三行倒数
  v1’ = (h21inv*u2+h22inv*v2+h23inv)*第三行倒数
  然后计算 和 真实 p1点 坐标的差值
   (u1-u2in1)*(u1-u2in1) + (v1-v2in1)*(v1-v2in1)   横纵坐标差值平方和
 * 
 【7】单应矩阵恢复  旋转矩阵 R 和平移向量t
 p2   =  H21 * p1   
 p2 = K( RP + t)  = KTP = H21 * KP  
 T =  K 逆 * H21*K
 
 【8】 基础矩阵 F求解
 *  * p2------> p1
 *                    f1   f2    f3      u1
 *   (u2 v2 1)    *   f4   f5    f6  *   v1    = 0  应该=0 不等于零的就是误差
 * 		      f7   f8    f9	  1
 * 	a1 = f1*u2 + f4*v2 + f7;
	b1 = f2*u2 + f5*v2 + f8;
	c1 =  f3*u2 + f6*v2 + f9;
       a1*u2+ b1*v2+ c1= 0
      一个点对 得到一个约束方程
       f1*u1*u2 + f2*v1*u2  + f3*u2 + f4*u1*v2  + f5*v1*v2 + f6*v2 +  f7*u1 + f8*v1 + f9 =0
       
     [  u1*u2   v1*u2   u2   u1*v2    v1*v2    v2  u1  v1 1 ] * [f1 f2 f3 f4 f5 f6 f7 f8 f9]转置  = 0
     
     8个点对 得到八个约束
     
     A *f = 0 求 f   奇异值分解得到F 基础矩阵 且其秩为2 需要再奇异值分解 后 取对角矩阵 秩为2 后在合成F
 
 * 【9】 基础矩阵 F 求变换误差
 *  * p2 ------> p1 
 *                    f11   f12    f13      u1
 *   (u2 v2 1)    *   f21   f22    f23  *   v1    = 0  应该=0 不等于零的就是误差
 * 		      f31   f32    f33 	    1
       p2转置 ×F 为 p2投影在 帧1中的极线 li = （a1 b1 c1）
 * 	a1 = f11*u2+f21*v2+f31;
	b1 = f12*u2+f22*v2+f32;
	c1 = f13*u2+f23*v2+f33;
	
 *  p1 应该在这条极限附近 求p1 到极线 l的距离 可以作为误差
 * 	 极线l：ax + by + c = 0
 * 	 (u,v)到l的距离为：d = |au+bv+c| / sqrt(a^2+b^2) 
 * 	 d^2 = |au+bv+c|^2/(a^2+b^2)	
 
	num1 = a1*u1 + b1*v1+ c1;// 应该等0
	num1*num1/(a1*a1+b1*b1); // 误差(点到直线的距离)
* 
 【10】  从基本矩阵恢复 旋转矩阵R 和 平移向量t
 计算 本质矩阵 E  =  K转置 * F  * K
 从本质矩阵恢复 旋转矩阵R 和 平移向量t
 恢复四种假设 并验证

 
 【11】2D-2D点三角化 得到对应的 三维点坐标
 平面二维点摄影矩阵到三维点  P1 = K × [I 0]     P2 = K * [R  t]
  kp1 = P1 * p3dC1       p3dC1  特征点匹配对 对应的 世界3维点
  kp2 = P2 * p3dC1  
  kp1 叉乘  P1 * p3dC1 =0
  kp2 叉乘  P2 * p3dC1 =0  
 p = ( x,y,1)
 其叉乘矩阵为
     //  叉乘矩阵 = [0  -1  y;
    //              1   0  -x; 
    //              -y   x  0 ]  
  一个方程得到两个约束
  对于第一行 0  -1  y; 会与P的三行分别相乘 得到四个值 与齐次3d点坐标相乘得到 0
  有 (y * P.row(2) - P.row(1) ) * D =0
      (-x *P.row(2) + P.row(0) ) * D =0 ===> (x *P.row(2) - P.row(0) ) * D =0
    两个方程得到 4个约束
    A × D = 0
    对A进行奇异值分解 求解线性方程 得到 D  （D是3维齐次坐标，需要除以第四个尺度因子 归一化）

    2也可转化到 相机归一化平面下的点  x1  x2
    p1 = k × [R1 t1] × D       k逆 × p1 =  [R1 t1] × D     x1 = T1 × D    x1叉乘x1 =  x1叉乘T1 × D = 0
    p2 = k × [ R2 t2]  × D     k逆 × p2 =  [R2 t2] × D     x2 = T2 × D    x2叉乘x2 =  x2叉乘T2 × D = 0
  
  
    
*/

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{
  /**
 * @brief 类构造函数  给定参考帧构造Initializer  单目相机初始化参考帧
 * 
 * 用reference frame来初始化，这个reference frame就是SLAM正式开始的第一帧
 * @param ReferenceFrame  参考帧
 * @param sigma                    测量误差
 * @param iterations              RANSAC迭代次数
 */
	Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
	{
	    mK = ReferenceFrame.mK.clone();// 相机内参数

	    mvKeys1 = ReferenceFrame.mvKeysUn;// 畸变校正后的 关键 点

	    mSigma = sigma;// 标准差
	    mSigma2 = sigma*sigma;// 方差
	    mMaxIterations = iterations;// 随机采样序列 最大迭代次数
	}
	
/**
 * @brief 类初始化函数 
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * @param CurrentFrame   当前帧       和 第一帧 参考帧 匹配 三角变换得到 3D点
 * @param vMatches12       当前帧 特征点的匹配信息
 * @param R21                     旋转矩阵 
 * @param t21                     平移矩阵  
 * @param vP3D                  恢复出的3D点
 * @param vbTriangulated 符合三角变换 的 3D点
 */
	bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
				    vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
	{
	    // Fill structures with current keypoints and matches with reference frame
	    // Reference Frame: 1,  Current Frame: 2
// Frame2  当前帧 畸变校正后的 关键 点
	    mvKeys2 = CurrentFrame.mvKeysUn;// 当前帧(2) 关键点

	    mvMatches12.clear();// 当前帧(2)  关键点 的匹配信息
	    mvMatches12.reserve(mvKeys2.size());
             // mvbMatched1记录每个特征点是否有匹配的特征点，
             // 这个变量后面没有用到，后面只关心匹配上的特征点    
	    mvbMatched1.resize(mvKeys1.size());// 匹配参考帧(1)关键点的匹配信息
	    
// 步骤1：根据 matcher.SearchForInitialization 得到的初始匹配点对，筛选后得到好的特征匹配点对
	    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
	    {
		if(vMatches12[i]>=0)// 帧2特征点 有匹配
		{
		    mvMatches12.push_back(make_pair(i,vMatches12[i]));
		    mvbMatched1[i]=true;
		}
		else
		    mvbMatched1[i]=false;// 未匹配到
	    }
	    
            // 匹配上的特征点的个数
	    const int N = mvMatches12.size();// 有效的 匹配点对 个数
	    
            // 新建一个容器vAllIndices，生成0到N-1的数作为特征点的索引
	    vector<size_t> vAllIndices;
	    vAllIndices.reserve(N);
	    vector<size_t> vAvailableIndices;

	    for(int i=0; i<N; i++)
	    {
		vAllIndices.push_back(i);
	    }

	    // Generate sets of 8 points for each RANSAC iteration
// 步骤2： 在所有匹配特征点对中随机选择8对特征匹配点对为一组，共选择mMaxIterations组
	    // 用于FindHomography和FindFundamental求解
	    // mMaxIterations:200	    
	    // 随机采样序列 最大迭代次数 随机序列 8点法 
	    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

	    DUtils::Random::SeedRandOnce(0);//随机数

	    for(int it=0; it<mMaxIterations; it++)
	    {
	      //随机参数 候选 点对
		vAvailableIndices = vAllIndices;//候选匹配点对 
		// Select a minimum set
		for(size_t j=0; j<8; j++)
		{ 
		    // 产生0到N-1的随机数
		    int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);// 随机数
		    // idx 表示哪一个索引对应的特征点被选中
		    int idx = vAvailableIndices[randi];// 对应的随机数
		    mvSets[it][j] = idx;//候选 匹配点对
		    
		    // randi对应的索引已经被选过了，从容器中删除
                     // randi对应的索引用最后一个元素替换，并删掉最后一个元素
		    vAvailableIndices[randi] = vAvailableIndices.back();
		    vAvailableIndices.pop_back();
		}
	    }
	    
// 步骤3：调用多线程分别用于计算fundamental matrix和homography
	    // Launch threads to compute in parallel a fundamental matrix and a homography
	    // 启动两个线程 分别计算 基本矩阵 F 和 单应矩阵
	    vector<bool> vbMatchesInliersH, vbMatchesInliersF;//内点标志  匹配点对是否在 计算出的 变换矩阵的 有效映射上
	    float SH, SF;// 最优变换矩阵 对应的 得分
	    cv::Mat H, F;//随机采样中计算得到 的 最优单应矩阵 H  和 基本矩阵 F
	    // 计算 单应矩阵 homograpy 并打分
	    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
	    // 计算 基础矩阵 fundamental matrix并打分
	    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

	    // Wait until both threads have finished
	    // 等待两个线程结束
	    threadH.join();
	    threadF.join();
// 步骤4：计算得分比例，选取某个模型	    
            //  从两个模型 H F 得分为 Sh   Sf 中选着一个 最优秀的 模型 的方法为
	    // Compute ratio of scores
	    float RH = SH/(SH+SF);// 计算 选着标志
	    
// 步骤5：根据评价得分，从单应矩阵H 或 基础矩阵F中恢复R,t
	    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
	    if(RH>0.40)// 更偏向于 平面  使用  单应矩阵恢复
		return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
	    else //if(pF_HF>0.6) // 偏向于非平面  使用 基础矩阵 恢复
		return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

	    return false;
	}

/**
 * @brief 计算单应矩阵   随机采样序列 8点  采用归一化的直接线性变换（normalized DLT）
 * 假设场景为平面情况下通过前两帧求取Homography矩阵(current frame 2 到 reference frame 1)
 * 在最大迭代次数内 调用 ComputeH21 计算  使用 CheckHomography 计算单应 得分
 * 并得到该模型的评分
 * 在最大迭代次数内 保留 最高得分的 单应矩阵
 * @param vbMatchesInliers     返回的 符合 变换的 匹配点 内点 标志
 * @param score                变换得分
 * @param H21                  单应矩阵
 */
	void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
	{
	    // Number of putative matches
	    const int N = mvMatches12.size();// 2中匹配的1中的点对 匹配点对总数

// 步骤1： // 将mvKeys1和mvKey2归一化到均值为0，一阶绝对矩为1，归一化矩阵分别为T1、T2	    
	    //2D-2D点对 求变换矩阵前先进行标准化  去均值点坐标 * 绝对矩倒数
	    //标准化矩阵  * 点坐标    =   标准化后的的坐标
	    // 点坐标    =    标准化矩阵 逆矩阵 * 标准化后的的坐标
	    // Normalize coordinates
	    vector<cv::Point2f> vPn1, vPn2;// 2d-2d点对
	    cv::Mat T1, T2;// 标准化矩阵
	    Normalize(mvKeys1,vPn1, T1);// 标准化点坐标  去均值点坐标 * 绝对矩倒数
	    Normalize(mvKeys2,vPn2, T2);// 
	    cv::Mat T2inv = T2.inv();// 标准化矩阵 逆矩阵

	    // Best Results variables
	    // 最终最佳的MatchesInliers与得分
	    score = 0.0;
	    vbMatchesInliers = vector<bool>(N,false);// 内点 标志

	    // Iteration variables
	    vector<cv::Point2f> vPn1i(8);// 随机 采样 8点对 
	    vector<cv::Point2f> vPn2i(8);
	    cv::Mat H21i, H12i;// 原点对 的 单应矩阵 //  H21i 原始点    p1 ----------------> p2 的单应
	    vector<bool> vbCurrentInliers(N,false);//当前随机点里的 内点
	    float currentScore;
	    
// 步骤2：随机采样序列迭代求解
	    // Perform all RANSAC iterations and save the solution with highest score
	    for(int it=0; it<mMaxIterations; it++)//在最大迭代次数内
	    {
		// Select a minimum set
//步骤3：随机8对点对
		for(size_t j=0; j<8; j++)
		{
		    int idx = mvSets[it][j];//随机数集合 总匹配点数范围内
		    vPn1i[j] = vPn1[mvMatches12[idx].first];
		    vPn2i[j] = vPn2[mvMatches12[idx].second];
		}           //       Hn                     T2逆*Hn*T1
// 步骤4：计算 单应矩阵        T1*p1  ----> T2*p2     p1 ----------------> p2
		cv::Mat Hn = ComputeH21(vPn1i,vPn2i);//  计算标准化后的点对 的 单应矩阵
		H21i = T2inv*Hn*T1;// 原始点    p1 ----------------> p2 的单应
		H12i = H21i.inv();// 原始点    p2 ----------------> p1 的单应
		
             // 计算单应 转换矩阵得分
		/*
		*  变换矩阵 评分 方法
		* SM=∑i( ρM( d2cr(xic,xir,M)   +   ρM( d2rc(xic,xir,M ) )
		*  d2cr 为 2D-2D 点对  通过哦转换矩阵的 对成转换误差
		* 
		* ρM 函数为 ρM(d^2)  = 0            当  d^2 > 阈值(单应矩阵时 为 5.99  基础矩阵时为 3.84)
		*                   最高分 - d^2    当  d^2 < 阈值
		*                                   最高分 均为 5.99 
		*/
// 步骤5：计算单应H的得分  有由 对应的匹配点对	的 对称的转换误差 求得	
		currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);
// 步骤6：保留最高得分 对应的 单应
		if(currentScore > score)//此次迭代 计算的单应H的得分较高
		{
		    H21 = H21i.clone();//保留较高得分的单应
		    vbMatchesInliers = vbCurrentInliers;//对应的匹配点对   
		    score = currentScore;// 最高的得分
		}
	    }
	}

// 计算基础矩阵   随机采样序列 8点  采用归一化的直接线性变换（normalized DLT）
// 在最大迭代次数内 调用 ComputeH21 计算  使用 CheckHomography 计算单应 得分
// 在最大迭代次数内 保留 最高得分的 单应矩阵
/**
 * @brief 计算基础矩阵
 *
 * 假设场景为非平面情况下通过前两帧求取Fundamental矩阵(current frame 2 到 reference frame 1),并得到该模型的评分
 */
	void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
	{
	    // Number of putative matches
	  // 总匹配点数
	    const int N = vbMatchesInliers.size();
/*
 *【1】2D-2D点对 求变换矩阵前先进行标准化  去均值点坐标 * 绝对矩倒数
 * 标准化矩阵  * 点坐标    =   标准化后的的坐标
 * 点坐标    =    标准化矩阵 逆矩阵 * 标准化后的的坐标
 */
	    // Normalize coordinates
	    vector<cv::Point2f> vPn1, vPn2;//  标准化后的的坐标
	    cv::Mat T1, T2;
	    Normalize(mvKeys1,vPn1, T1);// 标准化 去均值点坐标 * 绝对矩倒数
	    Normalize(mvKeys2,vPn2, T2);
	    cv::Mat T2t = T2.t();// 标准化矩阵 逆矩阵

	    // Best Results variables
	    score = 0.0;
	    vbMatchesInliers = vector<bool>(N,false);// 最优 基本矩阵变换  对应 的点对的标记  1是 内点  0 是野点

	    // Iteration variables
	    // 随机8对 点对
	    vector<cv::Point2f> vPn1i(8);
	    vector<cv::Point2f> vPn2i(8);
	    cv::Mat F21i;
	    vector<bool> vbCurrentInliers(N,false);//每次迭代 求解的 点对的标记  1是 内点  0 是野点
	    float currentScore;
 // 【2】随机采样序列迭代求解
	    // Perform all RANSAC iterations and save the solution with highest score
	    for(int it=0; it<mMaxIterations; it++)
	    {
		// Select a minimum set
        //【3】随机8对点对      
		for(int j=0; j<8; j++)
		{
		    int idx = mvSets[it][j];
		    vPn1i[j] = vPn1[mvMatches12[idx].first];
		    vPn2i[j] = vPn2[mvMatches12[idx].second];
		}
                             //       Fn                     T2逆*Fn*T1
    // 【4】计算 基础矩阵        T1*p1  ----> T2*p2     p1 ----------------> p2
		cv::Mat Fn = ComputeF21(vPn1i,vPn2i);
		F21i = T2t*Fn*T1;
             // 计算基础矩阵 F转换矩阵得分
		/*
		*  变换矩阵 评分 方法
		* SM=∑i( ρM( d2cr(xic,xir,M)   +   ρM( d2rc(xic,xir,M ) )
		*  d2cr 为 2D-2D 点对  通过哦转换矩阵的 对成转换误差
		* 
		* ρM 函数为 ρM(d^2)  = 0                  当  d^2 > 阈值(单应矩阵时 为 5.99  基础矩阵时为 3.84)
		*                                       最高分 - d^2    当  d^2 < 阈值
		*                                                            				   最高分 均为 5.99 
		*/
     // 【5】计算基础矩阵 F的得分  有由 对应的匹配点对	的 对称的转换误差 求得	
		currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);
     // 【6】保留最高得分 对应的 基础矩阵 F
		if(currentScore>score)
		{
		    F21 = F21i.clone();// 最优的 基础矩阵 F
		    vbMatchesInliers = vbCurrentInliers;//保持最优的 每次迭代 求解的 点对的标记  1是 内点  0 是野点
		    score = currentScore;//当前得分
		}
	    }
	}

// 计算单应矩阵  8对点对 每个点提供两个约束   A × h = 0 求h 奇异值分解 求 h
// // 通过svd进行最小二乘求解
// 参考   http://www.fengbing.net/
// |x'|     | h1 h2 h3 ||x|
// |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
// |1 |     | h7 h8 h9 ||1|
// 使用DLT(direct linear tranform)求解该模型
// x' = a H x 
// ---> (x') 叉乘 (H x)  = 0
// ---> Ah = 0
// A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
//     |-x -y -1  0  0  0 xx' yx' x'|
// 通过SVD求解Ah = 0，A'A最小特征值对应的特征向量即为解
/**
 * @brief 从特征点匹配求homography（normalized DLT）
 * 
 * @param  vP1 归一化后的点, in reference frame
 * @param  vP2 归一化后的点, in current frame
 * @return     单应矩阵
 * @see        Multiple View Geometry in Computer Vision - Algorithm 4.2 p109
 */
	cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
	{
	    const int N = vP1.size();// 8 点对
	    cv::Mat A(2*N,9,CV_32F);// 每个点 可以提供两个约束  单应为 3*3 9个 元素
/*
 *  1点 变成 2点   p2   =  H21 * p1
      u2         h1  h2  h3       u1
      v2  =      h4  h5  h6    *  v1
      1          h7  h8  h9       1   
      
     或是使用叉乘 得到0    * x = H y ，则对向量 x和Hy 进行叉乘为0，即：
					* | 0 -1  v2|    |h1 h2 h3|      |u1|     |0|
					* | 1  0 -u2| *  |h4 h5 h6| *    |v1| =  |0|
					* |-v2  u2 0|    |h7 h8 h9|      |1 |     |0|
      
      
      u2 = (h1*u1 + h2*v1 + h3) /( h7*u1 + h8*v1 + h9)
      v2 = (h4*u1 + h5*v1 + h6) /( h7*u1 + h8*v1 + h9)
    
      -((h4*u1 + h5*v1 + h6) - ( h7*u1*v2 + h8*v1*v2 + h9*v2))=0  式子为0  左侧加 - 号不变
        h1*u1 + h2*v1 + h3 - ( h7*u1*u2 + h8*v1*u2 + h9*u2)=0
        
        0    0   0  -u1  -v1  -1   u1*v2   v1*v2    v2
        u1 v1  1    0    0    0   -u1*u2  - v1*u2  -u2    ×(h1 h2 h3 h4 h5 h6 h7 h8 h9)转置  = 0
        
        8对点  约束 A 
        A × h = 0 求h   奇异值分解 A 得到 单元矩阵 H
 */	    
	    for(int i=0; i<N; i++)//8对点
	    {
		const float u1 = vP1[i].x;
		const float v1 = vP1[i].y;
		const float u2 = vP2[i].x;
		const float v2 = vP2[i].y;
		// 每个点对 两个而约束方程
		A.at<float>(2*i,0) = 0.0;
		A.at<float>(2*i,1) = 0.0;
		A.at<float>(2*i,2) = 0.0;
		A.at<float>(2*i,3) = -u1;
		A.at<float>(2*i,4) = -v1;
		A.at<float>(2*i,5) = -1;
		A.at<float>(2*i,6) = v2*u1;
		A.at<float>(2*i,7) = v2*v1;
		A.at<float>(2*i,8) = v2;
		
		A.at<float>(2*i+1,0) = u1;
		A.at<float>(2*i+1,1) = v1;
		A.at<float>(2*i+1,2) = 1;
		A.at<float>(2*i+1,3) = 0.0;
		A.at<float>(2*i+1,4) = 0.0;
		A.at<float>(2*i+1,5) = 0.0;
		A.at<float>(2*i+1,6) = -u2*u1;
		A.at<float>(2*i+1,7) = -u2*v1;
		A.at<float>(2*i+1,8) = -u2;
	    }
	    cv::Mat u,w,vt;
// A × h = 0 求h
	    // 在matlab中，[U,S,V]=svd(A)，其中U和V代表二个相互正交矩阵，而S代表一对角矩阵。 
	    //和QR分解法相同者， 原矩阵A不必为正方矩阵。
	    //使用SVD分解法的用途是解最小平方误差法和数据压缩。
	    // cv::SVDecomp(A,S,U,VT,SVD::FULL_UV);  //后面的FULL_UV表示把U和VT补充称单位正交方阵;
	    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);// 奇异值分解

	    return vt.row(8).reshape(0, 3);// v的最后一列
	}

// 通过svd进行最小二乘求解

// 8 对点 每个点 提供 一个约束	      
//8个点对 得到八个约束
//A *f = 0 求 f   奇异值分解 得到 f
/**
	*      构建基础矩阵的约束方程，给定一对点对应m=(u1,v1,1)T, m'=(u2,v2,1)T
	*  	   满足基础矩阵F   m'T F m=0,令F=(f_ij),则约束方程可以化简为：
	*  	    u2u1 f_11 + u2v1 f_12 + u2 f_13+v2u1f_21+v2v1f_22+v2f_23+u1f_31+v1f_32+f_33=0
	*  	    令f = (f_11,f_12,f_13,f_21,f_22,f_23,f_31,f_32,f_33)
	*  	    则(u2u1,u2v1,u2,v2u1,v2v1,v2,u1,v1,1)f=0;
	*  	    这样，给定N个对应点就可以得到线性方程组Af=0
	*  	    A就是一个N*9的矩阵，由于基础矩阵是非零的，所以f是一个非零向量，即
	*  	    线性方程组有非零解，另外基础矩阵的秩为2，重要的约束条件
	*/

// x'Fx = 0 整理可得：Af = 0
// A = | x'x x'y x' y'x y'y y' x y 1 |, f = | f1 f2 f3 f4 f5 f6 f7 f8 f9 |
// 通过SVD求解Af = 0，A'A最小特征值对应的特征向量即为解
/**
 * @brief 从特征点匹配求fundamental matrix（normalized 8点法）
 * @param  vP1 归一化后的点, in reference frame
 * @param  vP2 归一化后的点, in current frame
 * @return     基础矩阵
 * @see          Multiple View Geometry in Computer Vision - Algorithm 11.1 p282 (中文版 p191)
 */
	cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
	{
	    const int N = vP1.size();

	    cv::Mat A(N,9,CV_32F);
/*
 *  * p2------> p1
 *                    f1   f2    f3     u1
 *   (u2 v2 1)    *   f4   f5    f6  *  v1    = 0  应该=0 不等于零的就是误差
 * 		      f7   f8    f9	 1
 * 	a1 = f1*u2 + f4*v2 + f7;
	b1 = f2*u2 + f5*v2 + f8;
	c1 =  f3*u2 + f6*v2 + f9;
	
       a1*u1+ b1*v1+ c1= 0
      一个点对 得到一个约束方程
       f1*u1*u2 + f2*v1*u2  + f3*u2 + f4*u1*v2  + f5*v1*v2 + f6*v2 +  f7*u1 + f8*v1 + f9 =0
       
     [  u1*u2   v1*u2   u2   u1*v2    v1*v2    v2  u1  v1 1 ] * [f1 f2 f3 f4 f5 f6 f7 f8 f9]转置  = 0
     
     8个点对 得到八个约束
     
     A *f = 0 求 f   奇异值分解得到F 基础矩阵 且其秩为2 需要再奇异值分解 后 取对角矩阵 秩为2 在合成F
           
 */
	    for(int i=0; i<N; i++)
	    {
		const float u1 = vP1[i].x;
		const float v1 = vP1[i].y;
		const float u2 = vP2[i].x;
		const float v2 = vP2[i].y;

		A.at<float>(i,0) = u2*u1;
		A.at<float>(i,1) = u2*v1;
		A.at<float>(i,2) = u2;
		A.at<float>(i,3) = v2*u1;
		A.at<float>(i,4) = v2*v1;
		A.at<float>(i,5) = v2;
		A.at<float>(i,6) = u1;
		A.at<float>(i,7) = v1;
		A.at<float>(i,8) = 1;
	    }

	    cv::Mat u,w,vt;

	    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	    cv::Mat Fpre = vt.row(8).reshape(0, 3);// F 基础矩阵的秩为2 需要在分解 后 取对角矩阵 秩为2 在合成F

	    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	    w.at<float>(2)=0;//  基础矩阵的秩为2，重要的约束条件

	    return  u * cv::Mat::diag(w)  * vt;// 在合成F
	}

// 计算单应矩阵 得分
/*
 * 【3】变换矩阵 评分 方法
 *  SM=∑i( ρM( d2cr(xic,xir,M)   +   ρM( d2rc(xic,xir,M ) )
 *  d2cr 为 2D-2D 点对  通过哦转换矩阵的 对成转换误差
 * 
 *  ρM 函数为 ρM(d^2)  = 0         当  d^2 > 阈值(单应矩阵时 为 5.991  基础矩阵时为 3.84)
 *                   阈值 - d^2    当  d^2 < 阈值
 * 
 */

/**
 * @brief 对给定的homography matrix打分
 * 
 * @see
 * - Author's paper - IV. AUTOMATIC MAP INITIALIZATION （2）
 * - Multiple View Geometry in Computer Vision - symmetric transfer errors: 4.2.2 Geometric distance
 * - Multiple View Geometry in Computer Vision - model selection 4.7.1 RANSAC
 */
	float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
	{   
	    const int N = mvMatches12.size();// 总匹配点对数量
	    
    // |h11 h12 h13|
    // |h21 h22 h23|
    // |h31 h32 h33|
	    const float h11 = H21.at<float>(0,0);  //  p1  ----> p2
	    const float h12 = H21.at<float>(0,1);
	    const float h13 = H21.at<float>(0,2);
	    const float h21 = H21.at<float>(1,0);
	    const float h22 = H21.at<float>(1,1);
	    const float h23 = H21.at<float>(1,2);
	    const float h31 = H21.at<float>(2,0);
	    const float h32 = H21.at<float>(2,1);
	    const float h33 = H21.at<float>(2,2);
	    

    // |h11inv h12inv h13inv|
    // |h21inv h22inv h23inv|
    // |h31inv h32inv h33inv|
	    const float h11inv = H12.at<float>(0,0);
	    const float h12inv = H12.at<float>(0,1);
	    const float h13inv = H12.at<float>(0,2);
	    const float h21inv = H12.at<float>(1,0);
	    const float h22inv = H12.at<float>(1,1);
	    const float h23inv = H12.at<float>(1,2);
	    const float h31inv = H12.at<float>(2,0);
	    const float h32inv = H12.at<float>(2,1);
	    const float h33inv = H12.at<float>(2,2);

	    vbMatchesInliers.resize(N);// 匹配点对是否在 变换矩阵对于的 变换上  是否是内点

	    float score = 0;
            // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
	    const float th = 5.991;// 单应变换误差 阈值
	    //信息矩阵，方差平方的倒数
	    const float invSigmaSquare = 1.0/(sigma*sigma);//方差 倒数
	    
           // N对特征匹配点
	    for(int i=0; i<N; i++)//计算单应矩阵 变换 每个点对时产生 的 对称的转换误差
	    {
		bool bIn = true;
                // 关键点坐标
		const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
		const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];
/*
 * 
 * 1点 变成 2点
 u2         h11  h12  h13       u1
 v2  =      h21  h22  h23    *  v1
 1          h31  h32  h33        1   第三行 
 
 * 2 点 变成 1点
 u1‘        h11inv   h12inv   h13inv        u2
 v1’  =     h21inv   h22inv   h23inv     *  v2
 1          h31inv   h32inv   h33inv        1    第三行 h31inv*u2+h32inv*v2+h33inv 
 前两行 同除以 第三行 消去非零因子
  p2 由单应转换到 p1
  u1‘ = (h11inv*u2+h12inv*v2+h13inv)* 第三行倒数
  v1’ = (h21inv*u2+h22inv*v2+h23inv)*第三行倒数
  然后计算 和 真实 p1点 坐标的差值
   (u1-u2in1)*(u1-u2in1) + (v1-v2in1)*(v1-v2in1)   横纵坐标差值平方和
 */

// 步骤1： p2 由单应转换到 p1 距离误差  以及得分
		const float u1 = kp1.pt.x;
		const float v1 = kp1.pt.y;
		const float u2 = kp2.pt.x;
		const float v2 = kp2.pt.y;
		// Reprojection error in first image
		// x2in1 = H12*x2
		// 将图像2中的特征点单应到图像1中
		// |u1|     |h11inv h12inv h13inv||u2|
		// |v1| =   |h21inv h22inv h23inv||v2|
		// |1 |     |h31inv h32inv h33inv||1 |
		const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);//第三行倒数
		const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;// p2 由单应转换到 p1‘
		const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;
		const float squareDist1 = (u1-u2in1)*(u1-u2in1) + (v1-v2in1)*(v1-v2in1);// 横纵坐标差值平方和
		const float chiSquare1 = squareDist1*invSigmaSquare;// 根据方差归一化误差
		if(chiSquare1>th)//距离大于阈值  改点 变换的效果差
		    bIn = false;
		else
		    score += th - chiSquare1;// 阈值 - 距离差值 得到 得分，差值越小  得分越高
		    
// 步骤2：p1由单应转换到 p2 距离误差  以及得分
		// Reprojection error in second image
		// x1in2 = H21*x1   p1点 变成p2点 误差
		const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);//第三行倒数
		const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
		const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;
		// 计算重投影误差
		const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);// 计算重投影误差
		// 根据方差归一化误差
		const float chiSquare2 = squareDist2*invSigmaSquare;
		if(chiSquare2>th)
		    bIn = false;
		else
		    score += th - chiSquare2;
		if(bIn)
		    vbMatchesInliers[i]=true;// 是内点  误差较小
		else
		    vbMatchesInliers[i]=false;// 是野点 误差较大
	    }

	    return score;
	}

// 计算 基础矩阵 得分
// 和卡方分布的对应值比较，由此判定该点是否为内点。累计内点的总得分
// p2转置 * F * p1 =  0 
/*
 * p2 ------> p1 
 *                    f11   f12    f13     u1
 *   (u2 v2 1)    *   f21   f22    f23  *  v1    = 0应该=0 不等于零的就是误差
 * 		      f31   f32    f33 	    1
 * 	a1 = f11*u2+f21*v2+f31;
	b1 = f12*u2+f22*v2+f32;
	c1 = f13*u2+f23*v2+f33;
	num1 = a1*u1 + b1*v1+ c1;// 应该等0
	num1*num1/(a1*a1+b1*b1);// 误差
 */

/**
 * @brief 对给定的fundamental matrix打分
 * p2 转置 * F21 * p1 = 0 
 * F21 * p1为 帧1 关键点 p1在 帧2 上的极线 l1
 * 
 * 
 *  p2 应该在这条极限附近 求p2 到极线 l的距离 可以作为误差
 * 	 极线l：ax + by + c = 0
 * 	 (u,v)到l的距离为：d = |au+bv+c| / sqrt(a^2+b^2) 
 * 	 d^2 = |au+bv+c|^2/(a^2+b^2)
 * 
 * p2 转置 * F21 为帧2 关键点 p2在 帧1 上的极线 l2
 * 
 * 
 * @see
 * - Author's paper - IV. AUTOMATIC MAP INITIALIZATION （2）
 * - Multiple View Geometry in Computer Vision - symmetric transfer errors: 4.2.2 Geometric distance
 * - Multiple View Geometry in Computer Vision - model selection 4.7.1 RANSAC
 */
	float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
	{
	    const int N = mvMatches12.size();

	    const float f11 = F21.at<float>(0,0);
	    const float f12 = F21.at<float>(0,1);
	    const float f13 = F21.at<float>(0,2);
	    const float f21 = F21.at<float>(1,0);
	    const float f22 = F21.at<float>(1,1);
	    const float f23 = F21.at<float>(1,2);
	    const float f31 = F21.at<float>(2,0);
	    const float f32 = F21.at<float>(2,1);
	    const float f33 = F21.at<float>(2,2);

	    vbMatchesInliers.resize(N);

	    float score = 0;
	    
            // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
	    const float th = 3.841;
	    const float thScore = 5.991;
            //信息矩阵，方差平方的倒数
	    const float invSigmaSquare = 1.0/(sigma*sigma);

	    for(int i=0; i<N; i++)
	    {
		bool bIn = true;

		const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
		const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

		const float u1 = kp1.pt.x;
		const float v1 = kp1.pt.y;
		const float u2 = kp2.pt.x;
		const float v2 = kp2.pt.y;
//  p1 ------> p2 误差 得分-------------------------------
		// Reprojection error in second image
		
		// l2=F21 x1=(a2,b2,c2)
		// F21x1可以算出x1在图像中x2对应的线l
		const float a2 = f11*u1+f12*v1+f13;
		const float b2 = f21*u1+f22*v1+f23;
		const float c2 = f31*u1+f32*v1+f33;
		
                // x2应该在l这条线上:x2点乘l = 0 
		// 计算x2特征点到 极线 的距离：
		// 极线l：ax + by + c = 0
		// (u,v)到l的距离为：d = |au+bv+c| / sqrt(a^2+b^2) 
		// d^2 = |au+bv+c|^2/(a^2+b^2)
		const float num2 = a2*u2+b2*v2+c2;
		const float squareDist1 = num2*num2/(a2*a2+b2*b2);// 点到线的几何距离 的平方
		// 根据方差归一化误差
		const float chiSquare1 = squareDist1*invSigmaSquare;
		if(chiSquare1>th)
		    bIn = false;
		else
		    score += thScore - chiSquare1;//得分

		// Reprojection error in second image
		// l1 =x2转置 × F21=(a1,b1,c1)
//  p2 ------> p1 误差 得分-------------------------
		const float a1 = f11*u2+f21*v2+f31;
		const float b1 = f12*u2+f22*v2+f32;
		const float c1 = f13*u2+f23*v2+f33;
		const float num1 = a1*u1+b1*v1+c1;
		const float squareDist2 = num1*num1/(a1*a1+b1*b1);
		const float chiSquare2 = squareDist2*invSigmaSquare;
		if(chiSquare2>th)
		    bIn = false;
		else
		    score += thScore - chiSquare2;// 得分

		if(bIn)
		    vbMatchesInliers[i]=true;//内点  误差较小
		else
		    vbMatchesInliers[i]=false;// 野点  误差较大
	    }

	    return score;
	}


/*
 从基本矩阵恢复 旋转矩阵R 和 平移向量t
 计算 本质矩阵 E  =  K转置逆 * F  * K
 从本质矩阵恢复 旋转矩阵R 和 平移向量t
 恢复四种假设 并验证
理论参考 Result 9.19 in Multiple View Geometry in Computer Vision
 */
//                          |0 -1  0|
// E = U Sigma V'   let W = |1  0  0| 为RZ(90)  绕Z轴旋转 90度（x变成原来的y y变成原来的-x z轴没变）
//                          |0  0  1|
// 得到4个解 E = [R|t]
// R1 = UWV' R2 = UW'V' t1 = U3 t2 = -U3

/**
 * @brief 从基本矩阵 F 恢复R t
 * 
 * 度量重构
 * 1. 由Fundamental矩阵结合相机内参K，得到Essential矩阵: \f$ E = k转置F k \f$
 * 2. SVD分解得到R t
 * 3. 进行cheirality check, 从四个解中找出最合适的解
 * 
 * @see Multiple View Geometry in Computer Vision - Result 9.19 p259
 */	
	bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
				    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
	{
	    int N=0;
	    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
		if(vbMatchesInliers[i])// 是内点
		    N++;// 符合 基本矩阵 F的 内点数量

	    // Compute Essential Matrix from Fundamental Matrix
 // 步骤1： 计算 本质矩阵 E  =  K转置 * F  * K
	    cv::Mat E21 = K.t()*F21*K;

	    cv::Mat R1, R2, t;
	    // Recover the 4 motion hypotheses 四种运动假设
	    /*
// 步骤2：  从本质矩阵恢复 旋转矩阵R 和 平移向量t
	    *  对 本质矩阵E 进行奇异值分解   得到可能的解
	    * t = u * RZ(90) * u转置 
	    * R= u * RZ(90) * V转置 
	    * 组合情况有四种
	    */
	    
	    // 虽然这个函数对t有归一化，但并没有决定单目整个SLAM过程的尺度
            // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
	    DecomposeE(E21,R1,R2,t);  

	    cv::Mat t1=t;
	    cv::Mat t2=-t;

// 步骤3： 恢复四种假设 并验证 Reconstruct with the 4 hyphoteses and check
	    // 这4个解中只有一个是合理的，可以使用可视化约束来选择，
	    // 与单应性矩阵做sfm一样的方法，即将4种解都进行三角化，然后从中选择出最合适的解。
	    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
	    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
	    float parallax1,parallax2, parallax3, parallax4;

	    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
	    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
	    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
	    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

	    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

	    R21 = cv::Mat();
	    t21 = cv::Mat();
            // minTriangulated为可以三角化恢复三维点的个数
	    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

	    int nsimilar = 0;
	    if(nGood1>0.7*maxGood)
		nsimilar++;
	    if(nGood2>0.7*maxGood)
		nsimilar++;
	    if(nGood3>0.7*maxGood)
		nsimilar++;
	    if(nGood4>0.7*maxGood)
		nsimilar++;

	    // If there is not a clear winner or not enough triangulated points reject initialization
	    if(maxGood<nMinGood || nsimilar>1)
	    {// 四个结果中如果没有明显的最优结果，则返回失败
		return false;// 初始化失败
	    }

	    // If best reconstruction has enough parallax initialize
	    // 比较大的视差角  四种假设
	    if(maxGood==nGood1)
	    {
		if(parallax1>minParallax)
		{
		    vP3D = vP3D1;
		    vbTriangulated = vbTriangulated1;

		    R1.copyTo(R21);
		    t1.copyTo(t21);
		    return true;// 初始化成功
		}
	    }else if(maxGood==nGood2)
	    {
		if(parallax2>minParallax)
		{
		    vP3D = vP3D2;
		    vbTriangulated = vbTriangulated2;

		    R2.copyTo(R21);
		    t1.copyTo(t21);
		    return true;// 初始化成功
		}
	    }else if(maxGood==nGood3)
	    {
		if(parallax3>minParallax)
		{
		    vP3D = vP3D3;
		    vbTriangulated = vbTriangulated3;

		    R1.copyTo(R21);
		    t2.copyTo(t21);
		    return true;// 初始化成功
		}
	    }else if(maxGood==nGood4)
	    {
		if(parallax4>minParallax)
		{
		    vP3D = vP3D4;
		    vbTriangulated = vbTriangulated4;

		    R2.copyTo(R21);
		    t2.copyTo(t21);
		    return true;// 初始化成功
		}
	    }

	    return false;// 初始化失败
	}

	
/*
 从单应矩阵恢复 旋转矩阵R 和 平移向量t
 理论参考 
  // Faugeras et al, Motion and structure from motion in a piecewise planar environment. 
  International Journal of Pattern Recognition and Artificial Intelligence, 1988.
  
 https://hal.archives-ouvertes.fr/inria-00075698/document
 
p2   =  H21 * p1   
p2 = K( RP + t)  = KTP = H21 * KP  
T =  K 逆 * H21*K

在求得单应性变化H后，本文使用FAUGERAS的论文[1]的方法，提取8种运动假设。
这个方法通过可视化约束来测试选择合理的解。但是如果在低视差的情况下，
点云会跑到相机的前面或后面，测试就会出现错误从而选择一个错误的解。
文中使用的是直接三角化 8种方案，检查两个相机前面具有较少的重投影误差情况下，
在视图低视差情况下是否大部分云点都可以看到。如果没有一个解很合适，就不执行初始化，
重新从第一步开始。这种方法在低视差和两个交叉的视图情况下，初始化程序更具鲁棒性。
 */	
// H矩阵分解常见有两种方法：Faugeras SVD-based decomposition 和 Zhang SVD-based decomposition
// 参考文献：Motion and structure from motion in a piecewise plannar environment
// 这篇参考文献和下面的代码使用了Faugeras SVD-based decomposition算法

/**
 * @brief 从H恢复R t
 *
 * @see
 * - Faugeras et al, Motion and structure from motion in a piecewise planar environment. International Journal of Pattern Recognition and Artificial Intelligence, 1988.
 * - Deeper understanding of the homography decomposition for vision-based control
 */
	bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
			      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
	{
	    int N=0;
	    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
		if(vbMatchesInliers[i])
		    N++;//匹配点对 内点

	    // 8种运动假设  We recover 8 motion hypotheses using the method of Faugeras et al.
	    // Motion and structure from motion in a piecewise planar environment.
	    // International Journal of Pattern Recognition and Artificial Intelligence, 1988
	    
           // 因为特征点是图像坐标系，所以将H矩阵由相机坐标系换算到图像坐标系
	    cv::Mat invK = K.inv();
	    cv::Mat A = invK*H21*K;

	    cv::Mat U,w,Vt,V;
	    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
	    V=Vt.t();

	    float s = cv::determinant(U)*cv::determinant(Vt);

	    float d1 = w.at<float>(0);
	    float d2 = w.at<float>(1);
	    float d3 = w.at<float>(2);
	    
            // SVD分解的正常情况是特征值降序排列
	    if(d1/d2<1.00001 || d2/d3<1.00001)
	    {
		return false;// 初始化失败
	    }

	    vector<cv::Mat> vR, vt, vn;
	    vR.reserve(8);
	    vt.reserve(8);
	    vn.reserve(8);

	    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
            // 法向量n'= [x1 0 x3] 对应ppt的公式17
	    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
	    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
	    float x1[] = {aux1,aux1,-aux1,-aux1};
	    float x3[] = {aux3,-aux3,aux3,-aux3};

	    //case d'=d2
	    // 计算ppt中公式19
	    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

	    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
	    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};
	    // 计算旋转矩阵 R‘，计算ppt中公式18
	    //          | ctheta         0   -aux_stheta|         | aux1|
	    // Rp =     |    0               1       0  |  tp =   |  0     |
	    //          | aux_stheta  0    ctheta       |         |-aux3|

	    //          | ctheta          0    aux_stheta|          | aux1|
	    // Rp =     |    0            1       0      |  tp =    |  0  |
	    //          |-aux_stheta  0    ctheta        |          | aux3|

	    //          | ctheta         0    aux_stheta|         |-aux1|
	    // Rp =     |    0             1       0    |  tp =  |  0     |
	    //          |-aux_stheta  0    ctheta       |         |-aux3|

	    //          | ctheta         0   -aux_stheta|         |-aux1|
	    // Rp = |    0               1       0      |  tp =   |  0  |
	    //          | aux_stheta  0    ctheta       |          | aux3|

	    for(int i=0; i<4; i++)
	    {
		cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
		Rp.at<float>(0,0)=ctheta;
		Rp.at<float>(0,2)=-stheta[i];
		Rp.at<float>(2,0)=stheta[i];
		Rp.at<float>(2,2)=ctheta;

		cv::Mat R = s*U*Rp*Vt;
		vR.push_back(R);

		cv::Mat tp(3,1,CV_32F);
		tp.at<float>(0)=x1[i];
		tp.at<float>(1)=0;
		tp.at<float>(2)=-x3[i];
		tp*=d1-d3;
		
		// 这里虽然对t有归一化，并没有决定单目整个SLAM过程的尺度
		// 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
		cv::Mat t = U*tp;
		vt.push_back(t/cv::norm(t));

		cv::Mat np(3,1,CV_32F);
		np.at<float>(0)=x1[i];
		np.at<float>(1)=0;
		np.at<float>(2)=x3[i];

		cv::Mat n = V*np;
		if(n.at<float>(2)<0)
		    n=-n;
		vn.push_back(n);
	    }

	    //case d'=-d2
	    // 计算ppt中公式22
	    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

	    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
	    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};
	    
              // 计算旋转矩阵 R‘，计算ppt中公式21
	    for(int i=0; i<4; i++)
	    {
		cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
		Rp.at<float>(0,0)=cphi;
		Rp.at<float>(0,2)=sphi[i];
		Rp.at<float>(1,1)=-1;
		Rp.at<float>(2,0)=sphi[i];
		Rp.at<float>(2,2)=-cphi;

		cv::Mat R = s*U*Rp*Vt;
		vR.push_back(R);

		cv::Mat tp(3,1,CV_32F);
		tp.at<float>(0)=x1[i];
		tp.at<float>(1)=0;
		tp.at<float>(2)=x3[i];
		tp*=d1+d3;

		cv::Mat t = U*tp;
		vt.push_back(t/cv::norm(t));

		cv::Mat np(3,1,CV_32F);
		np.at<float>(0)=x1[i];
		np.at<float>(1)=0;
		np.at<float>(2)=x3[i];

		cv::Mat n = V*np;
		if(n.at<float>(2)<0)
		    n=-n;
		vn.push_back(n);
	    }


	    int bestGood = 0;
	    int secondBestGood = 0;    
	    int bestSolutionIdx = -1;
	    float bestParallax = -1;
	    vector<cv::Point3f> bestP3D;
	    vector<bool> bestTriangulated;

	    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
	    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
	    // d'=d2和d'=-d2分别对应8组(R t)
	    for(size_t i=0; i<8; i++)
	    {
		float parallaxi;
		vector<cv::Point3f> vP3Di;
		vector<bool> vbTriangulatedi;
		int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);
                // 保留最优的和次优的
		if(nGood>bestGood)
		{
		    secondBestGood = bestGood;
		    bestGood = nGood;
		    bestSolutionIdx = i;
		    bestParallax = parallaxi;
		    bestP3D = vP3Di;
		    bestTriangulated = vbTriangulatedi;
		}
		else if(nGood>secondBestGood)
		{
		    secondBestGood = nGood;
		}
	    }


	    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
	    {
		vR[bestSolutionIdx].copyTo(R21);
		vt[bestSolutionIdx].copyTo(t21);
		vP3D = bestP3D;
		vbTriangulated = bestTriangulated;

		return true;// 初始化成功
	    }

	    return false;// 初始化失败
	}

/*
 * 三角化得到3D点 
 *  *三角测量法 求解 两组单目相机  图像点深度
 * s1 * x1 = s2  * R * x2 + t
 * x1 x2 为两帧图像上 两点对 在归一化坐标平面上的坐标 k逆* p
 * s1  和 s2为两个特征点的深度 ，由于误差存在， s1 * x1 = s2  * R * x2 + t不精确相等
 * 常见的是求解最小二乘解，而不是零解
 *  s1 * x1叉乘x1 = s2 * x1叉乘* R * x2 + x1叉乘 t=0 可以求得x2
 * 
 */
/*
 平面二维点摄影矩阵到三维点  P1 = K × [I 0]    P2 = K * [R  t]
  kp1 = P1 * p3dC1       p3dC1  特征点匹配对 对应的 世界3维点
  kp2 = P2 * p3dC1  
  kp1 叉乘  P1 * p3dC1 =0
  kp2 叉乘  P2 * p3dC1 =0  
 p = ( x,y,1)
 其叉乘矩阵为
     //  叉乘矩阵 = [0  -1  y;
    //              1   0  -x; 
    //              -y   x  0 ]  
  一个方程得到两个约束
  对于第一行 0  -1  y; 会与P的三行分别相乘 得到四个值 与齐次3d点坐标相乘得到 0
  有 (y * P.row(2) - P.row(1) ) * D =0
      (-x *P.row(2) + P.row(0) ) * D =0 ===> (x *P.row(2) - P.row(0) ) * D =0
    两个方程得到 4个约束
    A × D = 0
    对A进行奇异值分解 求解线性方程 得到 D  （D是3维齐次坐标，需要除以第四个尺度因子 归一化）
 */
// Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
// x' = P'X  x = PX
// 它们都属于 x = aPX模型
//                            |X|
// |x|     |p1 p2   p3  p4|   |Y|     |x|    |--p0--||.|
// |y| = a |p5 p6   p7  p8|   |Z| ===>|y| = a|--p1--||X|
// |z|     |p9 p10 p11 p12|   |1|     |z|    |--p2--||.|
// 采用DLT的方法：x叉乘PX = 0
// |yp2 -  p1|      |0|
// |p0  -  xp2| X = |0|
// |xp1 - yp0|      |0|
// 两个点:
// |yp2   -  p1  |     |0|
// |p0    -  xp2 | X = |0| ===> AX = 0
// |y'p2' -  p1' |     |0|
// |p0'   - x'p2'|     |0|
// 变成程序中的形式：
// |xp2  - p0 |     |0|
// |yp2  - p1 | X = |0| ===> AX = 0
// |x'p2'- p0'|     |0|
// |y'p2'- p1'|     |0|
/**
 * @brief 给定投影矩阵P1,P2和图像上的点kp1,kp2，从而恢复3D坐标
 *
 * @param kp1 特征点, in reference frame
 * @param kp2 特征点, in current frame
 * @param P1  投影矩阵P1
 * @param P2  投影矩阵P２
 * @param x3D 三维点
 * @see       Multiple View Geometry in Computer Vision - 12.2 Linear triangulation methods p312
 */
	void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
	{
	  // 在DecomposeE函数和ReconstructH函数中对t有归一化
	  // 这里三角化过程中恢复的3D点深度取决于 t 的尺度，
	  // 但是这里恢复的3D点并没有决定单目整个SLAM过程的尺度
	  // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
	    cv::Mat A(4,4,CV_32F);

	    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
	    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
	    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
	    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

	    cv::Mat u,w,vt;
	    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
	    x3D = vt.row(3).t();
	    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);//  转换成非齐次坐标  归一化
	}

	// 对  2D 点对进行标准化 
/*
 *【0】2D-2D点对 求变换矩阵前先进行标准化  去 均值后再除以绝对矩
* 单目初始化特征点 归一化 坐标均值为0  一阶绝对矩为1
* mean_x  =  sum( ui) / N   mean_y =  sum(vi)/N
* 绝对矩  mean_x_dev = sum（abs(ui - mean_x)）/ N     mean_y_dev = sum（abs(vi - mean_y)）/ N 
*
*绝对矩倒数 sX = 1/mean_x_dev     sY = 1/mean_y_dev
* 
* 标准化后的点坐标
* u = (ui - mean_x) × sX
* v =  (vi - mean_y) * sY 
* 
* 标准化矩阵   其逆矩阵×标准化点 得到原始坐标 
*      用于 计算变换矩阵后  从原始坐标计算对称的转换误差 来计算变换矩阵得分
* T = sX   0    -mean_x * sX
*     0   sY   -mean_y * sY
*     0    0         1
* 
* 标准化矩阵  * 点坐标    =   标准化后的的坐标
*       ui         ui × sX - mean_x * sX  = (ui - mean_x) × sX       u
*  T ×  vi    =    vi  × sY - mean_y * sY  = (vi - mean_y) × sY   =   v
*       1               1              				      1
* 
* * 点坐标    =    标准化矩阵 逆矩阵 * 标准化后的的坐标
* 
 */	
/**
 * ＠brief 归一化特征点到同一尺度（作为normalize DLT的输入）
 *
 * [x' y' 1]' = T * [x y 1]' \n
 * 归一化后x', y'的均值为0，sum(abs(x_i'-0))=1，sum(abs((y_i'-0))=1
 * 
 * @param vKeys             特征点在图像上的坐标
 * @param vNormalizedPoints 特征点归一化后的坐标
 * @param T                 将特征点归一化的矩阵
 */
	void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
	{

	    const int N = vKeys.size();// 点总数
	    vNormalizedPoints.resize(N);//标准化后的点
	    
	    float meanX = 0;//横坐标均值
	    float meanY = 0;//纵坐标均值
	    for(int i=0; i<N; i++)
	    {
		meanX += vKeys[i].pt.x;// 横坐标之和
		meanY += vKeys[i].pt.y;//纵坐标之和
	    }
	    meanX = meanX/N;//横坐标均值
	    meanY = meanY/N;//纵坐标均值

	    float meanDevX = 0;//绝对矩
	    float meanDevY = 0;//绝对矩
	    
           // 将所有vKeys点减去中心坐标，使x坐标和y坐标均值分别为0
	    for(int i=0; i<N; i++)
	    {
		vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;// 去均值点坐标
		vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;// 
		meanDevX += fabs(vNormalizedPoints[i].x);// 总绝对矩
		meanDevY += fabs(vNormalizedPoints[i].y);
	    }
	    meanDevX = meanDevX/N;//均值绝对矩
	    meanDevY = meanDevY/N;

	    float sX = 1.0/meanDevX;
	    float sY = 1.0/meanDevY;
	    
           // 将x坐标和y坐标分别进行尺度缩放，使得x坐标和y坐标的一阶绝对矩分别为1
	    for(int i=0; i<N; i++)
	    {
	      // 标注化后的点坐标
		vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;// 去均值点坐标 * 绝对矩倒数
		vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
	    }
    // |sX  0  -meanx*sX|
    // |0   sY -meany*sY|
    // |0   0       1   |	    
            // 标准化矩阵
            // 标准化矩阵  * 点坐标    =   标准化后的的坐标
            //  点坐标    =    标准化矩阵 逆矩阵 * 标准化后的的坐标
	    T = cv::Mat::eye(3,3,CV_32F);
	    T.at<float>(0,0) = sX;
	    T.at<float>(1,1) = sY;
	    T.at<float>(0,2) = -meanX*sX;
	    T.at<float>(1,2) = -meanY*sY;
	}


/*
 * 检查求得的R t 是否符合
 * 接受 R,t ，一组成功的匹配。最后给出的结果是这组匹配中有多少匹配是
 * 能够在这组 R,t 下正确三角化的（即 Z都大于0），并且输出这些三角化之后的三维点。

如果三角化生成的三维点 Z小于等于0，且三角化的“前方交会角”（余弦是 cosParallax）不会太小，
那么这个三维点三角化错误，舍弃。

通过了 Z的检验，之后将这个三维点分别投影到两张影像上，
计算投影的像素误差，误差大于2倍中误差，舍弃。

 */
/**
 * @brief 进行cheirality check，从而进一步找出F分解后最合适的解
 */
	int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
			      const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
			      const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
	{
	    // Calibration parameters
	   // 校正参数
	    const float fx = K.at<float>(0,0);
	    const float fy = K.at<float>(1,1);
	    const float cx = K.at<float>(0,2);
	    const float cy = K.at<float>(1,2);

	    vbGood = vector<bool>(vKeys1.size(),false);
	    vP3D.resize(vKeys1.size());// 对应的三维点

	    vector<float> vCosParallax;
	    vCosParallax.reserve(vKeys1.size());

	    // Camera 1 Projection Matrix K[I|0]
// 步骤1：得到一个相机的投影矩阵
            // 以第一个相机的光心作为世界坐标系	    
	    // 相机1  变换矩阵 在第一幅图像下 的变换矩阵  Pc1  =   Pw  =  T1 * Pw      T1 = [I|0]
	    // Pp1  = K *  Pc1 = K * T1 * Pw  =   [K|0] *Pw  = P1 × Pw
	    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
	    K.copyTo(P1.rowRange(0,3).colRange(0,3));
            // 第一个相机的光心在世界坐标系下的坐标
	    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);// 相机1原点 000
	    
// 步骤2：得到第二个相机的投影矩阵
	    // Camera 2 Projection Matrix K[R|t]
	    // 相机2  变换矩阵  Pc2  =   Pw  =  T2 * Pw      T2 = [R|t]
	    // Pp2  = K *  Pc2 = K * T2 * Pw  =  K* [R|t] *Pw  = P2 × Pw 
	    cv::Mat P2(3,4,CV_32F);
	    R.copyTo(P2.rowRange(0,3).colRange(0,3));
	    t.copyTo(P2.rowRange(0,3).col(3));
	    P2 = K*P2;
            // 第二个相机的光心在世界坐标系下的坐标
	    cv::Mat O2 = -R.t()*t;//相机2原点  R逆 * - t  R 为正交矩阵  逆 = 转置

	    int nGood=0;

	    for(size_t i=0, iend=vMatches12.size();i<iend;i++)// 每一个匹配点对
	    {
		if(!vbMatchesInliers[i])// 离线点  非内点
		    continue;
               // kp1和kp2是匹配特征点
		const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
		const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
		cv::Mat p3dC1;
		
// 步骤3：利用三角法恢复三维点p3dC1
		// kp1 = P1 * p3dC1     kp2 = P2 * p3dC1   
		Triangulate(kp1,kp2,P1,P2,p3dC1);

		if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
		{// 求出的3d点坐标 值有效
		    vbGood[vMatches12[i].first]=false;
		    continue;
		}
		
// 步骤4：计算视差角余弦值
		// Check parallax
		cv::Mat normal1 = p3dC1 - O1;
		float dist1 = cv::norm(normal1);

		cv::Mat normal2 = p3dC1 - O2;
		float dist2 = cv::norm(normal2);

		float cosParallax = normal1.dot(normal2)/(dist1*dist2);
		
 // 步骤5：判断3D点是否在两个摄像头前方
		// Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
	   // 步骤5.1：3D点深度为负，在第一个摄像头后方，淘汰
		if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
		    continue;
		
           // 步骤5.2：3D点深度为负，在第二个摄像头后方，淘汰
		// Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
		cv::Mat p3dC2 = R*p3dC1+t;

		if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
		    continue;
		
// 步骤6：计算重投影误差
		// Check reprojection error in first image
		// 计算3D点在第一个图像上的投影误差
		float im1x, im1y;
		float invZ1 = 1.0/p3dC1.at<float>(2);
		im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
		im1y = fy*p3dC1.at<float>(1)*invZ1+cy;
		float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);
		
         // 步骤6.1：重投影误差太大，跳过淘汰
                 // 一般视差角比较小时重投影误差比较大
		if(squareError1>th2)
		    continue;
		
               // 计算3D点在第二个图像上的投影误差
		// Check reprojection error in second image
		float im2x, im2y;
		float invZ2 = 1.0/p3dC2.at<float>(2);
		im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
		im2y = fy*p3dC2.at<float>(1)*invZ2+cy;
		float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

         // 步骤6.2：重投影误差太大，跳过淘汰
                 // 一般视差角比较小时重投影误差比较大
		if(squareError2>th2)
		    continue;
		
         // 步骤7：统计经过检验的3D点个数，记录3D点视差角
		vCosParallax.push_back(cosParallax);
		vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
		//nGood++;

		if(cosParallax<0.99998){
		    vbGood[vMatches12[i].first]=true;
		  // WYW  20180130 修改
		  nGood++;
		 }
	    }
	    
// 步骤8：得到3D点中较大的视差角
	    if(nGood>0)
	    {
		sort(vCosParallax.begin(),vCosParallax.end());// 从小到大排序
		
	      // trick! 排序后并没有取最大的视差角
	      // 取一个较大的视差角
		size_t idx = min(50,int(vCosParallax.size()-1));
		parallax = acos(vCosParallax[idx])*180/CV_PI;
	    }
	    else
		parallax=0;

	    return nGood;
	}

	
/*
 * 从本质矩阵恢复 旋转矩阵R 和 平移向量t
 *  对 本质矩阵E 进行奇异值分解   得到可能的解
 * t = u * RZ(90) * u转置 
 * R= u * RZ(90) * V转置 
 * 组合情况有四种
 */

/**
 * @brief 分解Essential矩阵
 * 
 * F矩阵通过结合内参可以得到Essential矩阵，分解E矩阵将得到4组解 \n
 * 这4组解分别为[R1,t],[R1,-t],[R2,t],[R2,-t]
 * @param E  Essential Matrix
 * @param R1 Rotation Matrix 1
 * @param R2 Rotation Matrix 2
 * @param t  Translation
 * @see Multiple View Geometry in Computer Vision - Result 9.19 p259
 */
	void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
	{
   // 【1】对 本质矩阵E 进行奇异值分解  
	    cv::Mat u,w,vt;
	    cv::SVD::compute(E,w,u,vt);// 其中u和v代表二个相互正交矩阵，而w代表一对角矩阵
	    
	    // 对 t 有归一化，但是这个地方并没有决定单目整个SLAM过程的尺度
	    // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
	    u.col(2).copyTo(t);
	    t=t/cv::norm(t);
// 沿着Z轴旋转 90度得到的旋转矩阵（逆时针为正方向）
// z 轴还是 原来的z轴   y轴变成原来的 x 轴的负方向   x轴变成原来的y轴
// 所以 旋转矩阵  为 0  -1   0
//		    1   0   0
//		    0   0   1
// 沿着Z轴旋转- 90度	  
// z 轴还是 原来的z轴   y轴变成原来的 x 轴   x轴变成原来的y轴的负方向
// 所以 旋转矩阵  为 0   1   0  为上 旋转矩阵的转置矩阵
//		    -1    0   0
//		    0   0   1    
	    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
	    W.at<float>(0,1)=-1;
	    W.at<float>(1,0)=1;
	    W.at<float>(2,2)=1;

	    R1 = u*W*vt;
	    if(cv::determinant(R1)<0)
		R1=-R1;

	    R2 = u*W.t()*vt;
	    if(cv::determinant(R2)<0)
		R2=-R2;
	}

} //namespace ORB_SLAM
