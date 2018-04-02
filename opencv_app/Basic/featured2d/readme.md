      #############################################################
# 2D Features framework (feature2d module)  特征检测
      ##############################################################

# 【1】Harris角点  cornerHarris()

        算法基本思想是使用一个固定窗口在图像上进行任意方向上的滑动，
        比较滑动前与滑动后两种情况，窗口中的像素灰度变化程度，
        如果存在任意方向上的滑动，都有着较大灰度变化，
        那么我们可以认为该窗口中存在角点。

        图像特征类型:
            边缘 （Edges   物体边缘）
            角点 (Corners  感兴趣关键点（ interest points） 边缘交叉点 )
            斑点(Blobs  感兴趣区域（ regions of interest ） 交叉点形成的区域 )


        为什么角点是特殊的?
            因为角点是两个边缘的连接点(交点)它代表了两个边缘变化的方向上的点。
            图像梯度有很高的变化。这种变化是可以用来帮助检测角点的。


        G = SUM( W(x,y) * [I(x+u, y+v) -I(x,y)]^2 )

         [u,v]是窗口的偏移量
         (x,y)是窗口内所对应的像素坐标位置，窗口有多大，就有多少个位置
         w(x,y)是窗口函数，最简单情形就是窗口内的所有像素所对应的w权重系数均为1。
                     设定为以窗口中心为原点的二元正态分布

        泰勒展开（I(x+u, y+v) 相当于 导数）
        G = SUM( W(x,y) * [I(x,y) + u*Ix + v*Iy - I(x,y)]^2)
          = SUM( W(x,y) * (u*u*Ix*Ix + v*v*Iy*Iy))
          = SUM(W(x,y) * [u v] * [Ix^2   Ix*Iy] * [u 
                            Ix*Iy  Iy^2]     v] )
          = [u v]  * SUM(W(x,y) * [Ix^2   Ix*Iy] ) * [u  应为 [u v]为常数 可以拿到求和外面
                             Ix*Iy  Iy^2]      v]    
          = [u v] * M * [u
                   v]
        则计算 det(M)   矩阵M的行列式的值  取值为一个标量，写作det(A)或 | A |  矩阵表示的空间的单位面积/体积/..
               trace(M) 矩阵M的迹         矩阵M的对角线元素求和，用字母T来表示这种算子，他的学名叫矩阵的迹

        M的两个特征值为 lamd1  lamd2

        det(M)    = lamd1 * lamd2
        trace(M) = lamd1 + lamd2

        R = det(M)  -  k*(trace(M))^2 
        其中k是常量，一般取值为0.04~0.06，
        R大于一个阈值的话就认为这个点是 角点

        因此可以得出下列结论：
        >特征值都比较大时，即窗口中含有角点
        >特征值一个较大，一个较小，窗口中含有边缘
        >特征值都比较小，窗口处在平坦区域

        https://blog.csdn.net/woxincd/article/details/60754658

# 【2】 Shi-Tomasi 算法 goodFeaturesToTrack()
        是Harris 算法的改进。
        Harris 算法最原始的定义是将矩阵 M 的行列式值与 M 的迹相减，
        再将差值同预先给定的阈值进行比较。

        后来Shi 和Tomasi 提出改进的方法，
        若两个特征值中较小的一个大于最小阈值，则会得到强角点。
        M 对角化>>> M的两个特征值为 lamd1  lamd2

        R = mini(lamd1,lamd2) > 阈值 认为是角点


# 【3】FAST角点检测算法  ORB特征检测中使用的就是这种角点检测算法
              FAST(src_gray, keyPoints,thresh);
        周围区域灰度值 都较大 或 较小

              若某像素与其周围邻域内足够多的像素点相差较大，则该像素可能是角点。

        该算法检测的角点定义为在像素点的周围邻域内有足够多的像素点与该点处于不同的区域。
        应用到灰度图像中，即有足够多的像素点的灰度值大于该点的灰度值或者小于该点的灰度值。

        p点附近半径为3的圆环上的16个点，
        一个思路是若其中有连续的12个点的灰度值与p点的灰度值差别超过某一阈值，
        则可以认为p点为角点。

        这一思路可以使用机器学习的方法进行加速。
        对同一类图像，例如同一场景的图像，可以在16个方向上进行训练，
        得到一棵决策树，从而在判定某一像素点是否为角点时，
        不再需要对所有方向进行检测，
        而只需要按照决策树指定的方向进行2-3次判定即可确定该点是否为角点。

        std::vector<KeyPoint> keyPoints; 
        //fast.detect(src_gray, keyPoints);  // 检测角点
        FAST(src_gray, keyPoints,thresh);

# 【4】 使用cornerEigenValsAndVecs()函数和cornerMinEigenVal()函数自定义角点检测函数。
        过自定义 R的计算方法和自适应阈值 来定制化检测角点

        计算 M矩阵
        计算判断矩阵 R

        设置自适应阈值

        阈值大小为 判断矩阵 最小值和最大值之间 百分比
        阈值为 最小值 + （最大值-最小值）× 百分比
        百分比 = myHarris_qualityLevel/max_qualityLevel

# 【5】亚像素级的角点检测
        如果对角点的精度有更高的要求，可以用cornerSubPix()函数将角点定位到子像素，
        从而取得亚像素级别的角点检测效果。

        使用cornerSubPix()函数在goodFeaturesToTrack()的角点检测基础上将角点位置精确到亚像素级别

        常见的亚像素级别精准定位方法有三类：
          1. 基于插值方法
          2. 基于几何矩寻找方法
          3. 拟合方法 - 比较常用

        拟合方法中根据使用的公式不同可以分为
          1. 高斯曲面拟合与
          2. 多项式拟合等等。

        以高斯拟合为例:

          窗口内的数据符合二维高斯分布
          Z = n / (2 * pi * 西格玛^2) * exp(-P^2/(2*西格玛^2))
          P = sqrt( (x-x0)^2 + (y-y0)^2)

          x,y   原来 整数点坐标
          x0,y0 亚像素补偿后的 坐标 需要求取

          ln(Z) = n0 + x0/(西格玛^2)*x +  y0/(西格玛^2)*y - 1/(2*西格玛^2) * x^2 - 1/(2*西格玛^2) * y^2
            n0 +            n1*x + n2*y +             n3*x^2 +              n3 * y^2

          对窗口内的像素点 使用最小二乘拟合 得到上述 n0 n1 n2 n3
            则 x0 = - n1/(2*n3)
               y0 = - n2/(2*n3)


【6】斑点检测原理 SIFT  SURF
 
	SIFT定位算法关键步骤的说明 
	http://www.cnblogs.com/ronny/p/4028776.html

        SIFT原理与源码分析 https://blog.csdn.net/xiaowei_cqu/article/details/8069548

	该算法大概可以归纳为三步：1）高斯差分金字塔的构建；2）特征点的搜索；3）特征描述。

	    DoG尺度空间构造（Scale-space extrema detection）
	    关键点搜索与定位（Keypoint localization）
	    方向赋值（Orientation assignment）
	    关键点描述（Keypoint descriptor）
	    OpenCV实现：特征检测器FeatureDetector
	    SIFT中LoG和DoG的比较

	SURF算法与源码分析、上  加速鲁棒特征（SURF）
	www.cnblogs.com/ronny/p/4045979.html
        https://blog.csdn.net/abcjennifer/article/details/7639681

	通过在不同的尺度上利用积分图像可以有效地计算出近似Harr小波值，
	简化了二阶微分模板的构建，搞高了尺度空间的特征检测的效率。
	在以关键点为中心的3×3×3像素邻域内进行非极大值抑制，
	最后通过对斑点特征进行插值运算，完成了SURF特征点的精确定位。

	而SURF特征点的描述，则也是充分利用了积分图，用两个方向上的Harr小波模板来计算梯度，
	然后用一个扇形对邻域内点的梯度方向进行统计，求得特征点的主方向。
      
      
      // SURF放在另外一个包的xfeatures2d里边了，在github.com/Itseez/opencv_contrib 这个仓库里。
	// 按说明把这个仓库编译进3.0.0就可以用了。
	opencv2中SurfFeatureDetector、SurfDescriptorExtractor、BruteForceMatcher在opencv3中发生了改变。
	具体如何完成特征点匹配呢？示例如下：

	//寻找关键点
	int minHessian = 700;
	Ptr<SURF>detector = SURF::create(minHessian);
	detector->detect( srcImage1, keyPoint1 );
	detector->detect( srcImage2, keyPoints2 );

	//绘制特征关键点
	Mat img_keypoints_1; Mat img_keypoints_2;
	drawKeypoints( srcImage1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	drawKeypoints( srcImage2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	//显示效果图
	imshow("特征点检测效果图1", img_keypoints_1 );
	imshow("特征点检测效果图2", img_keypoints_2 );

	//计算特征向量
	Ptr<SURF>extractor = SURF::create();
	Mat descriptors1, descriptors2;
	extractor->compute( srcImage1, keyPoint1, descriptors1 );
	extractor->compute( srcImage2, keyPoints2, descriptors2 );

	//使用BruteForce进行匹配
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	std::vector< DMatch > matches;
	matcher->match( descriptors1, descriptors2, matches );

	//绘制从两个图像中匹配出的关键点
	Mat imgMatches;
	drawMatches( srcImage1, keyPoint1, srcImage2, keyPoints2, matches, imgMatches );//进行绘制
	//显示
	imshow("匹配图", imgMatches );


	3.x的特征检测:

	    算法：SURF,SIFT,BRIEF,FREAK 
	    类：cv::xfeatures2d::SURF

	    cv::xfeatures2d::SIFT
	    cv::xfeatures::BriefDescriptorExtractor
	    cv::xfeatures2d::FREAK
	    cv::xfeatures2d::StarDetector

	    需要进行以下几步

	    加入opencv_contrib
	    包含opencv2/xfeatures2d.hpp
	    using namepsace cv::xfeatures2d
	    使用create(),detect(),compute(),detectAndCompute()


【7】二进制字符串特征描述子
	注意到在两种角点检测算法里，我们并没有像SIFT或SURF那样提到特征点的描述问题。
	事实上，特征点一旦检测出来，无论是斑点还是角点描述方法都是一样的，
	可以选用你认为最有效的特征描述子。

	比较有代表性的就是 浮点型特征描述子(sift、surf 欧氏距离匹配)和
	二进制字符串特征描述子 (字符串汉明距离 匹配 )。

	1 浮点型特征描述子(sift、surf 欧氏距离匹配):
		像SIFT与SURF算法里的，用梯度统计直方图来描述的描述子都属于浮点型特征描述子。
		但它们计算起来，算法复杂，效率较低.

		SIFT特征采用了128维的特征描述子，由于描述子用的浮点数，所以它将会占用512 bytes的空间。
		类似地，对于SURF特征，常见的是64维的描述子，它也将占用256bytes的空间。
		如果一幅图像中有1000个特征点（不要惊讶，这是很正常的事），
		那么SIFT或SURF特征描述子将占用大量的内存空间，对于那些资源紧张的应用，
		尤其是嵌入式的应用，这样的特征描述子显然是不可行的。而且，越占有越大的空间，
		意味着越长的匹配时间。

		我们可以用PCA、LDA等特征降维的方法来压缩特征描述子的维度。
		还有一些算法，例如LSH，将SIFT的特征描述子转换为一个二值的码串，
		然后这个码串用汉明距离进行特征点之间的匹配。这种方法将大大提高特征之间的匹配，
		因为汉明距离的计算可以用异或操作然后计算二进制位数来实现，在现代计算机结构中很方便。

	2 二进制字符串特征描述子 (字符串汉明距离 匹配 ):
		如BRIEF。后来很多二进制串描述子ORB，BRISK，FREAK等都是在它上面的基础上的改进。

	【A】 BRIEF:  Binary Robust Independent Elementary Features
	  http://www.cnblogs.com/ronny/p/4081362.html

		它需要先平滑图像，然后在特征点周围选择一个Patch，在这个Patch内通过一种选定的方法来挑选出来nd个点对。
		然后对于每一个点对(p,q)，我们来比较这两个点的亮度值，
		如果I(p)>I(q)则这个点对生成了二值串中一个的值为1，
		如果I(p)<I(q)，则对应在二值串中的值为-1，否则为0。
		所有nd个点对，都进行比较之间，我们就生成了一个nd长的二进制串。

		对于nd的选择，我们可以设置为128，256或512，这三种参数在OpenCV中都有提供，
		但是OpenCV中默认的参数是256，这种情况下，非匹配点的汉明距离呈现均值为128比特征的高斯分布。
		一旦维数选定了，我们就可以用汉明距离来匹配这些描述子了。

		对于BRIEF，它仅仅是一种特征描述符，它不提供提取特征点的方法。
		所以，如果你必须使一种特征点定位的方法，如FAST、SIFT、SURF等。
		这里，我们将使用CenSurE方法来提取关键点，对BRIEF来说，CenSurE的表现比SURF特征点稍好一些。
		总体来说，BRIEF是一个效率很高的提取特征描述子的方法，
		同时，它有着很好的识别率，但当图像发生很大的平面内的旋转。


		关于点对的选择：

		设我们在特征点的邻域块大小为S×S内选择nd个点对(p,q)，Calonder的实验中测试了5种采样方法：

			1）在图像块内平均采样；
			2）p和q都符合(0,1/25 * S^2)的高斯分布；
			3）p符合(0,1/25 * S^2)的高斯分布，而q符合(0,1/100 *S^2)的高斯分布；
			4）在空间量化极坐标下的离散位置随机采样
			5）把p固定为(0,0)，q在周围平均采样

	【B】BRISK算法
	 	BRISK算法在特征点检测部分没有选用FAST特征点检测，而是选用了稳定性更强的AGAST算法。
		在特征描述子的构建中，BRISK算法通过利用简单的像素灰度值比较，
		进而得到一个级联的二进制比特串来描述每个特征点，这一点上原理与BRIEF是一致的。
		BRISK算法里采用了邻域采样模式，即以特征点为圆心，构建多个不同半径的离散化Bresenham同心圆，
		然后再每一个同心圆上获得具有相同间距的N个采样点。	

	【C】ORB算法 Oriented FAST and Rotated BRIEF
 	   http://www.cnblogs.com/ronny/p/4083537.html
		ORB算法使用FAST进行特征点检测，然后用BREIF进行特征点的特征描述，
		但是我们知道BRIEF并没有特征点方向的概念，所以ORB在BRIEF基础上引入了方向的计算方法，
		并在点对的挑选上使用贪婪搜索算法，挑出了一些区分性强的点对用来描述二进制串。

          	通过构建高斯金字塔 来实现 尺度不变性
	  	利用灰度质心法     来实现 记录方向
			灰度质心法假设角点的灰度与质心之间存在一个偏移，这个向量可以用于表示一个方向。
	【D】FREAK算法 Fast Retina KeyPoint，即快速视网膜关键点。
	 	根据视网膜原理进行点对采样，中间密集一些，离中心越远越稀疏。
		并且由粗到精构建描述子，穷举贪婪搜索找相关性小的。
		42个感受野，一千对点的组合，找前512个即可。这512个分成4组，
		前128对相关性更小，可以代表粗的信息，后面越来越精。匹配的时候可以先看前16bytes，
		即代表精信息的部分，如果距离小于某个阈值，再继续，否则就不用往下看了。

【8】KAZE非线性尺度空间 特征

	基于非线性尺度空间的KAZE特征提取方法以及它的改进AKATE
	https://blog.csdn.net/chenyusiyuan/article/details/8710462

	KAZE是日语‘风’的谐音，寓意是就像风的形成是空气在空间中非线性的流动过程一样，
	KAZE特征检测是在图像域中进行非线性扩散处理的过程。

	传统的SIFT、SURF等特征检测算法都是基于 线性的高斯金字塔 进行多尺度分解来消除噪声和提取显著特征点。
	但高斯分解是牺牲了局部精度为代价的，容易造成边界模糊和细节丢失。

	非线性的尺度分解有望解决这种问题，但传统方法基于正向欧拉法（forward Euler scheme）
	求解非线性扩散（Non-linear diffusion）方程时迭代收敛的步长太短，耗时长、计算复杂度高。

	由此，KAZE算法的作者提出采用加性算子分裂算法(Additive Operator Splitting, AOS)
	来进行非线性扩散滤波，可以采用任意步长来构造稳定的非线性尺度空间。


	非线性扩散滤波
		Perona-Malik扩散方程:
			具体地，非线性扩散滤波方法是将图像亮度（L）在不同尺度上的变化视为某种形式的
			流动函数（flow function）的散度（divergence），可以通过非线性偏微分方程来描述：
		AOS算法:
			由于非线性偏微分方程并没有解析解，一般通过数值分析的方法进行迭代求解。
			传统上采用显式差分格式的求解方法只能采用小步长，收敛缓慢。

	KAZE特征检测与描述

	KAZE特征的检测步骤大致如下：
	1) 首先通过AOS算法和可变传导扩散（Variable  Conductance  Diffusion）（[4,5]）方法来构造非线性尺度空间。
	2) 检测感兴趣特征点，这些特征点在非线性尺度空间上经过尺度归一化后的Hessian矩阵行列式是局部极大值（3×3邻域）。
	3) 计算特征点的主方向，并且基于一阶微分图像提取具有尺度和旋转不变性的描述向量。

	特征点检测
	KAZE的特征点检测与SURF类似，是通过寻找不同尺度归一化后的Hessian局部极大值点来实现的。
