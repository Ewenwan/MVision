//-----------------------------------【程序说明】----------------------------------------------  
//      程序名称:：《【OpenCV入门教程之十二】OpenCV边缘检测：Canny算子,Sobel算子,Laplace算子,Scharr滤波器合辑合辑》   
//      开发所用IDE版本：Visual Studio 2010  
//----------------------------------------------------------------------------------------------  
//关于边缘检测     Canny算子,Sobel算子,Laplace算子以及Scharr滤波器 
//步骤  -> 滤波 -> 增强 -> 检测
//滤波 边缘检测的算法主要是基于图像强度的一阶和二阶导数，但导数通常对噪声很敏感，因此必须采用滤波器来改善与噪声有关的边缘检测器的性能。常见的滤波方法主要有高斯滤波，
//增强 增强边缘的基础是确定图像各点邻域强度的变化值。增强算法可以将图像灰度点邻域强度值有显著变化的点凸显出来。在具体编程实现时，可通过计算梯度幅值来确定
//检测 经过增强的图像，往往邻域中有很多点的梯度值比较大，而在特定的应用中，这些点并不是我们要找的边缘点，所以应该采用某种方法来对这些点进行取舍。实际工程中，常用的方法是通过阈值化方法来检测
//另外，需要注意，下文中讲到的Laplace算子，sobel算子和Scharr算子都是带方向的，所以，示例中我们分别写了X方向,Y方向和最终合成的的效果图。
//Canny边缘检测算子是John F.Canny于 1986 年开发出来的一个多级边缘检测算法。
//=======滤波------>>>>>>>>>使用高斯平滑滤波器卷积降噪
//=======增强1------>>>>>>>>>Sobel滤波器 计算像素 梯度幅值和方向   (分别作用于 x 和 y 方向) 梯度方向近似到四个可能角度之一(一般为0, 45, 90, 135)   sqrt(Gx^+Gy^)
//=======增强2------>>>>>>>>>非极大值抑制(对于梯度赋值)。 这一步排除非边缘像素， 仅仅保留了一些细线条(候选边缘)。
//=======检测------>>>>>>>>>滞后阈值。最后一步，Canny 使用了滞后阈值，滞后阈值需要两个阈值(高阈值和低阈值)
//=======Ⅰ 如果某一像素位置的幅值超过 高 阈值, 该像素被保留为边缘像素。
//=======Ⅱ 如果某一像素位置的幅值小于 低 阈值, 该像素被排除。
//=======Ⅲ 如果某一像素位置的幅值在两个阈值之间,若该像素 连接到 一个高于高阈值的像素 时则被保留。
//tips：对于Canny函数的使用，推荐的高低阈值比在2:1到3:1之间
//函数解析
//void Canny(InputArray image,OutputArray edges, double threshold1, double threshold2, int apertureSize=3,bool L2gradient=false )  
//第一个参数，InputArray类型的image，输入图像，即源图像，填Mat类的对象即可，且需为单通道8位图像。
//第二个参数，OutputArray类型的edges，输出的边缘图，需要和源图片有一样的尺寸和类型。
//第三个参数，double类型的threshold1，第一个滞后性阈值。
//第四个参数，double类型的threshold2，第二个滞后性阈值。
//第五个参数，int类型的apertureSize，表示应用Sobel算子的孔径大小，其有默 认值3。  计算梯度方向和赋值的模板大小
//第六个参数，bool类型的L2gradient，一个计算图像梯度幅值的标识，有默认值false。
//需要注意的是，这个函数阈值1和阈值2两者的小者用于边缘连接，而大者用来控制强边缘的初始段，推荐的高低阈值比在2:1到3:1之间。
//sobel算子的计算过程 
//==============梯度赋值计算简化   |Gx|+|Gy|
//函数详解
//Sobel函数使用扩展的 Sobel 算子，来计算一阶、二阶、三阶或混合图像差分。
//void Sobel (  InputArray src,  OutputArray dst,  int ddepth,  int dx,  int dy,  int ksize=3,  double scale=1,  double delta=0,  int borderType=BORDER_DEFAULT );
/*
第一个参数，InputArray 类型的src，为输入图像，填Mat类型即可。
第二个参数，OutputArray类型的dst，即目标图像，函数的输出参数，需要和源图片有一样的尺寸和类型。
第三个参数，int类型的ddepth，输出图像的深度，支持如下src.depth()和ddepth的组合：
   若src.depth() = CV_8U, 取ddepth =-1/CV_16S/CV_32F/CV_64F
   若src.depth() = CV_16U/CV_16S, 取ddepth =-1/CV_32F/CV_64F
   若src.depth() = CV_32F, 取ddepth =-1/CV_32F/CV_64F
   若src.depth() = CV_64F, 取ddepth = -1/CV_64F
第四个参数，int类型dx，x 方向上的差分阶数。
第五个参数，int类型dy，y方向上的差分阶数。
第六个参数，int类型ksize，有默认值3，表示Sobel核的大小;必须取1，3，5或7。
第七个参数，double类型的scale，计算导数值时可选的缩放因子，默认值是1，表示默认情况下是没有应用缩放的。我们可以在文档中查阅getDerivKernels的相关介绍，来得到这个参数的更多信息。
第八个参数，double类型的delta，表示在结果存入目标图（第二个参数dst）之前可选的delta值，有默认值0。
第九个参数， int类型的borderType，我们的老朋友了（万年是最后一个参数），边界模式，默认值为BORDER_DEFAULT。这个参数可以在官方文档中borderInterpolate处得到更详细的信息。
*/


//-----------------------------------【头文件包含部分】---------------------------------------  
//      描述：包含程序所依赖的头文件  
//---------------------------------------------------------------------------------------------- 
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
  
//-----------------------------------【命名空间声明部分】--------------------------------------  
//-----------------------------------------------------------------------------------------------   
using namespace cv;  
  
 //-----------------------------------【全局变量声明部分】--------------------------------------  
//      描述：全局变量声明  
//-----------------------------------------------------------------------------------------------  
//原图，原图的灰度版，目标图  
Mat g_srcImage, g_srcGrayImage,g_dstImage;  
  
//Canny边缘检测相关变量  
Mat g_cannyDetectedEdges;  
int g_cannyLowThreshold=1;//TrackBar位置参数    
  
//Sobel边缘检测相关变量  
Mat g_sobelGradient_X, g_sobelGradient_Y;  
Mat g_sobelAbsGradient_X, g_sobelAbsGradient_Y;  
int g_sobelKernelSize=1;//内核   
 
//Laplacian边缘检测相关变量
Mat src_gas, src_gas_gray,g_LaplacianDetectedEdges; 
int g_LaplacianKernelSize=1;

//Scharr滤波器相关变量  
Mat g_scharrGradient_X, g_scharrGradient_Y;  
Mat g_scharrAbsGradient_X, g_scharrAbsGradient_Y;  
  
  
//-----------------------------------【全局函数声明部分】--------------------------------------  
//      描述：全局函数声明  
//-----------------------------------------------------------------------------------------------  
static void ShowHelpText( );  
static void on_Canny(int, void*);//Canny边缘检测窗口滚动条的回调函数  
static void on_Sobel(int, void*);//Sobel边缘检测窗口滚动条的回调函数
static void on_Laplacian(int, void*);//Sobel边缘检测窗口滚动条的回调函数
void Scharr( );//封装了Scharr边缘检测相关代码的函数  
  
  
//-----------------------------------【main( )函数】--------------------------------------------  
//      描述：控制台应用程序的入口函数，我们的程序从这里开始  
//-----------------------------------------------------------------------------------------------  
int main( int argc, char** argv )  
{  
    //改变console字体颜色  
    system("color 2E");    
  
    //显示欢迎语  
    ShowHelpText();  
  
    //载入原图   "E:\\VSworkSpace\\HelloOpenCV\\x64\\Debug\\2.jpg"
    g_srcImage = imread("E:\\VSworkSpace\\ImgEdgeDetection\\baby.jpg");  
    if( !g_srcImage.data ) { printf("Oh，no，读取srcImage错误~！ \n"); return false; }  
  
    //显示原始图  
    namedWindow("[1]【原始图】");  
    imshow("[1]【原始图】", g_srcImage);  
  
    // 创建与src同类型和大小的矩阵(dst)  
    g_dstImage.create( g_srcImage.size(), g_srcImage.type() );  
  
    // 将原图像转换为灰度图像  
    cvtColor( g_srcImage, g_srcGrayImage, COLOR_BGR2GRAY );    //BGR 变成 gray 灰色
  
    // 创建显示窗口  
    namedWindow( "[2]【效果图】Canny边缘检测", WINDOW_AUTOSIZE );  // WINDOW_AUTOSIZE=1
    namedWindow( "[3]【效果图】Sobel边缘检测", WINDOW_AUTOSIZE ); 
	namedWindow( "[4]【效果图】Laplacian边缘检测", WINDOW_AUTOSIZE ); 
  
    // 创建trackbar  
    createTrackbar( "参数值：", "[2]【效果图】Canny边缘检测", &g_cannyLowThreshold, 120, on_Canny );  
    createTrackbar( "参数值：", "[3]【效果图】Sobel边缘检测", &g_sobelKernelSize, 3, on_Sobel );  
	createTrackbar( "参数值：", "[4]【效果图】Laplacian边缘检测", &g_LaplacianKernelSize, 3, on_Laplacian );
  
    // 调用回调函数  
    on_Canny(g_cannyLowThreshold, 0);  
    on_Sobel(g_sobelKernelSize, 0);  
    on_Laplacian(g_LaplacianKernelSize, 0);

    //调用封装了Scharr边缘检测代码的函数  
    Scharr( );  
  
    //轮询获取按键信息，若按下Q，程序退出  
    while((char(waitKey(1)) != 'q')) {}  
  
    return 0;  

}  
  
  
//-----------------------------------【ShowHelpText( )函数】----------------------------------  
//      描述：输出一些帮助信息  
//----------------------------------------------------------------------------------------------  
static void ShowHelpText()  
{  
    //输出一些帮助信息  
    printf( "\n\n\t嗯。运行成功，请调整滚动条观察图像效果~\n\n"  
        "\t按下“q”键时，程序退出~!\n"  
        "\n\n\t\t\t\t " );  
}  
  
  
//-----------------------------------【on_Canny( )函数】----------------------------------  
//      描述：Canny边缘检测窗口滚动条的回调函数  
//-----------------------------------------------------------------------------------------------  
void on_Canny(int, void*)  
{  
	//对灰度图进行滤波降噪
    // 先使用 3x3内核来降噪  
	// box filter 方框滤波函数进行降噪
    blur( g_srcGrayImage, g_cannyDetectedEdges, Size(3,3) );  
  
    // 运行我们的Canny算子
	//            原图              目标图              滞后阈值    下限      上限       梯度计算模板大小
    Canny( g_cannyDetectedEdges, g_cannyDetectedEdges, g_cannyLowThreshold, g_cannyLowThreshold*3, 3 );  
  
    //先将g_dstImage内的所有Scalar元素设置为0   
    g_dstImage = Scalar::all(0);  
  
    //使用Canny算子输出的边缘图g_cannyDetectedEdges作为掩码，来将原图g_srcImage拷到目标图g_dstImage中  
    g_srcImage.copyTo( g_dstImage, g_cannyDetectedEdges);  
  
    //显示效果图  
    imshow( "[2]【效果图】Canny边缘检测", g_dstImage );  
}  
  
  
  
//-----------------------------------【on_Sobel( )函数】----------------------------------  
//      描述：Sobel边缘检测窗口滚动条的回调函数  
//-----------------------------------------------------------------------------------------  
void on_Sobel(int, void*)  
{  
    // 求 X方向梯度                               x  y     核 大小
    Sobel( g_srcImage, g_sobelGradient_X, CV_16S, 1, 0, (2*g_sobelKernelSize+1), 1, 1, BORDER_DEFAULT );  
    convertScaleAbs( g_sobelGradient_X, g_sobelAbsGradient_X );//计算绝对值，并将结果转换成8位  
  
    // 求Y方向梯度                                x  y 
    Sobel( g_srcImage, g_sobelGradient_Y, CV_16S, 0, 1, (2*g_sobelKernelSize+1), 1, 1, BORDER_DEFAULT );  
    convertScaleAbs( g_sobelGradient_Y, g_sobelAbsGradient_Y );//计算绝对值，并将结果转换成8位  
  
    // 合并梯度  
    addWeighted( g_sobelAbsGradient_X, 0.5, g_sobelAbsGradient_Y, 0.5, 0, g_dstImage );  
  
    //显示效果图  
    imshow("[3]【效果图】Sobel边缘检测", g_dstImage);   
  
}  
//-----------------------------------【on_Laplacian( )函数】----------------------------------  
//      描述：Laplacian边缘检测窗口滚动条的回调函数  
//-----------------------------------------------------------------------------------------  
void on_Laplacian(int, void*)  
{  
	//之前已经转化成 灰度图
	//[1] 使用高斯滤波消除噪声 
	GaussianBlur( g_srcImage, src_gas, Size(3,3), 0, 0, BORDER_DEFAULT );

	cvtColor( src_gas, src_gas_gray,  COLOR_RGB2GRAY ); 

    //[2] 使用Laplace函数
    Laplacian( src_gas_gray, g_LaplacianDetectedEdges, CV_16S, (2*g_LaplacianKernelSize+1), 1, 0, BORDER_DEFAULT );
    //[3] 计算绝对值，并将结果转换成8位  
    convertScaleAbs( g_LaplacianDetectedEdges, g_LaplacianDetectedEdges );   
    //[4] 显示效果图  
    imshow( "[4]【效果图】Laplacian边缘检测", g_LaplacianDetectedEdges); 
   
  
}  
  
//-----------------------------------【Scharr( )函数】----------------------------------  
//      描述：封装了Scharr边缘检测相关代码的函数  
//-----------------------------------------------------------------------------------------  
void Scharr( )  
{  
    // 求 X方向梯度  
    Scharr( g_srcImage, g_scharrGradient_X, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT );  
    convertScaleAbs( g_scharrGradient_X, g_scharrAbsGradient_X );//计算绝对值，并将结果转换成8位  
  
    // 求Y方向梯度  
    Scharr( g_srcImage, g_scharrGradient_Y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT );  
    convertScaleAbs( g_scharrGradient_Y, g_scharrAbsGradient_Y );//计算绝对值，并将结果转换成8位  
  
    // 合并梯度  
    addWeighted( g_scharrAbsGradient_X, 0.5, g_scharrAbsGradient_Y, 0.5, 0, g_dstImage );  
  
    //显示效果图  
    imshow("[5]【效果图】Scharr滤波器", g_dstImage);   
}  
