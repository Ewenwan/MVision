/*
基于距离变换和
分水岭算法的图像分割


图像分割的定义

根据灰度、颜色、纹理和形状等特征，把图像分成若干个特定的、
具有独特性质的区域，这些特征在同一区域内呈现出相似性，
而在不同区域间呈现出明显的差异性，并提出感兴趣目标的技术和过程。
 它是由图像处理到图像分析的关键步骤。从数学角度来看，
图像分割是将数字图像划分成互不相交的区域的过程。
图像分割的过程也是一个标记过程，即把属于同一区域的像素赋予相同的编号。

图像分割方法：基于阈值的分割方法、基于边缘的分割方法、基于区域的分割方法、基于特定理论的分割方法等。

图像分割方法概述如下：

【1】、基于阈值的分割方法
	阈值法的基本思想是基于图像的灰度特征来计算一个或多个灰度阈值，
	并将图像中每个像素的灰度值与阈值相比较，最后将像素根据比较结果分到合适的类别中。
	因此，该类方法最为关键的一步就是按照某个准则函数来求解最佳灰度阈值。

【2】基于边缘的分割方法
	所谓边缘是指图像中两个不同区域的边界线上连续的像素点的集合，
	是图像局部特征不连续性的反映，体现了灰度、颜色、纹理等图像特性的突变。
	通常情况下，基于边缘的分割方法指的是基于灰度值的边缘检测，
	它是建立在边缘灰度值会呈现出阶跃型或屋顶型变化这一观测基础上的方法。

         阶跃型边缘两边像素点的灰度值存在着明显的差异，而屋顶型边缘则位于灰度值上升或下降的转折处。
	对于阶跃状边缘常用微分算子进行边缘检测，其位置对应一阶导数的极值点，对应二阶导数的过零点(零交叉点)。
	常用的一阶微分算子有Roberts算子、Prewitt算子和Sobel算子，
	二阶微分算子有Laplace算子和Kirsh算子等。
	在实际中各种微分算子常用小区域模板来表示，微分运算是利用模板和图像卷积来实现。
	这些算子对噪声敏感，只适合于噪声较小不太复杂的图像。

【3】基于区域的分割方法
	此类方法是将图像按照相似性准则分成不同的区域，
	主要包括种子区域生长法、区域分裂合并法和分水岭法等几种类型。

	【A】种子区域生长法是从一组代表不同生长区域的种子像素开始，
		接下来将种子像素邻域里符合条件的像素合并到种子像素所代表的生长区域中，
		并将新添加的像素作为新的种子像素继续合并过程，直到找不到符合条件的新像素为止。
		该方法的关键是 选择合适的初始种子像素以及 合理的生长准则。

	【B】区域分裂合并法（Gonzalez，2002）的基本思想是首先将图像任意分成若干互不相交的区域，
		然后再按照相关准则对这些区域进行分裂或者合并从而完成分割任务，
		该方法既适用于 灰度图像分割 也适用于 纹理图像分割。

	【C】分水岭法（Meyer，1990）是一种基于拓扑理论的数学形态学的分割方法，
		其基本思想是把图像看作是 测地学上的拓扑地貌，图像中每一点像素的灰度值表示该点的 海拔高度，
		每一个局部极小值及其影响区域称为 集水盆，而集水盆的边界则形成分水岭。
		该算法的实现可以模拟成洪水淹没的过程，图像的最低点首先被淹没，然后水逐渐淹没整个山谷。
		当水位到达一定高度的时候将会溢出，这时在水溢出的地方修建堤坝，重复这个过程直到整个图像上的点全部被淹没，
		这时所建立的一系列堤坝就成为分开各个盆地的分水岭。
		分水岭算法对微弱的边缘有着良好的响应，但图像中的噪声会使分水岭算法产生过分割的现象。

【4】基于图论的分割方法
	此类方法把 图像分割问题 与 图的最小割（min cut）问题 相关联。
	首先将图像映射为 带权无向图G=<V，E>，图中每个节点N∈V对应于图像中的每个像素，
	每条边∈E连接着一对相邻的像素，边的权值表示了相邻像素之间在灰度、颜色或纹理方面的非负相似度。
	而对图像的一个分割s就是对图的一个剪切，被分割的每个区域C∈S对应着图中的一个子图。
	而分割的最优原则就是使划分后的子图在内部保持相似度最大，而子图之间的相似度保持最小。
	基于图论的分割方法的本质就是移除特定的边，将图划分为若干子图从而实现分割。
	目前所了解到的基于图论的方法有GraphCut、GrabCut和Random Walk等。

【5】基于能量泛函的分割方法
	该类方法主要指的是活动轮廓模型（active contour model）以及在其基础上发展出来的算法，
	其基本思想是使用连续曲线来表达目标边缘，并定义一个能量泛函使得其自变量包括边缘曲线，
	因此分割过程就转变为求解能量泛函的最小值的过程，一般可通过求解函数对应的欧拉(Euler．Lagrange)方程来实现，
	能量达到最小时的曲线位置就是目标的轮廓所在。按照模型中曲线表达形式的不同，
	活动轮廓模型可以分为两大类：
	参数活动轮廓模型（parametric active contour model）和
	几何活动轮廓模型（geometric active contour model）。

	【A】参数活动轮廓模型
		是基于Lagrange框架，直接以曲线的参数化形式来表达曲线，
		最具代表性的是由Kasset a1(1987)所提出的Snake模型。该类模型在早期的生物图像分割领域得到了成功的应用，
		但其存在着分割结果受初始轮廓的设置影响较大以及难以处理曲线拓扑结构变化等缺点，
		此外其能量泛函只依赖于曲线参数的选择，与物体的几何形状无关，这也限制了其进一步的应用。

	【B】几何活动轮廓模型
		的曲线运动过程是基于曲线的几何度量参数而非曲线的表达参数，
		因此可以较好地处理拓扑结构的变化，并可以解决参数活动轮廓模型难以解决的问题。
		而水平集（Level Set）方法（Osher，1988）的引入，则极大地推动了几何活动轮廓模型的发展，
		因此几何活动轮廓模型一般也可被称为水平集方法。

Opencv中相应的API

（1）距离变换：计算源图像的每个像素到最近的零像素的距离.

distanceTransform(

InputArray src, //8位单通道(二进制)源图像

OutputArray dst, //输出具有计算距离的图像,8位或32位浮点的单通道图像，大小与src相同.

OutputArray labels, //输出二维数组标签labels（离散维诺Voronoi图）

int distanceType, //距离类型 = DIST_L1/DIST_L2

int maskSize, //距离变换的掩膜大小，DIST_MASK_3(maskSize = 3x3),最新的支持DIST_MASK_5(mask = 5x5)，推荐3x3

int labelType=DIST_LABEL_CCOMP //要生成的标签数组的类型

)

（2）分水岭：用分水岭算法执行基于标记的图像分割

watershed(

InputArray image, //输入8位3通道图像（输入锐化原图8位3通道）

InputOutputArray markers // 输入或输出32位单通道的标记，和图像一样大小。（输入高峰轮廓标记）

)
*/


#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
int main(int argc, char** argv)
{
// 加载源图像
    string imageName("../../common/data/apple.jpeg"); // 图片文件名路径（默认值）
    if( argc > 1)
    {
        imageName = argv[1];//如果传递了文件 就更新
    }

    Mat src = imread( imageName );
    if( src.empty() )
     { 
        cout <<  "can't load image " << endl;
	return -1;  
     }

    // 显示原图像
    imshow("Source Image", src);
    // 白色像素 (255,255,255) 变为黑色 (0,0,0)
    for( int x = 0; x < src.rows; x++ ) {
      for( int y = 0; y < src.cols; y++ ) {
          if ( src.at<Vec3b>(x, y) == Vec3b(255,255,255) ) {
            src.at<Vec3b>(x, y)[0] = 0;
            src.at<Vec3b>(x, y)[1] = 0;
            src.at<Vec3b>(x, y)[2] = 0;
          }
        }
    }
    // 显示 黑背景图像
    imshow("Black Background Image", src);
    // 锐化图像的 核   二阶导 类似 拉普拉斯算子
    Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1); 
   // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    Mat sharp = src; // 复制图像
    filter2D(sharp, imgLaplacian, CV_32F, kernel);// 滤波锐化
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;


    // 转换成8 位 灰度图
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // 显示锐化灰度图像
    // imshow( "Laplace Filtered Image", imgLaplacian );
    imshow( "New Sharped Image", imgResult );

    src = imgResult; // 复制结果图像
    // 创建二值化图像 阈值 40
    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
    threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    imshow("Binary Image", bw);

// 距离变换
    Mat dist;
// 计算源图像的每个像素到最近的零像素的距离.
    distanceTransform(bw, dist, CV_DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
// 归一化距离矩阵
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects

// >0.4 的 设置为 1 阈值操作
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(dist, dist, kernel1);//膨胀 白色区域 增长
    imshow("Peaks", dist);

    // Create the CV_8U version of the distance image
    // It is needed for findContours()
// 在转换成 8位
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

// 寻找轮廓
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);//标记像素 属于那个领域

// 绘制不同领域的像素
    for (size_t i = 0; i < contours.size(); i++)
    drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);

    // Draw the background marker
    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    imshow("Markers", markers*10000);

//执行分水岭算法  Perform the watershed algorithm
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);//二值化
//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
                                  // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst.at<Vec3b>(i,j) = colors[index-1];
            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }
    // Visualize the final image
    imshow("Final Result", dst);
    waitKey(0);
    return 0;
}
