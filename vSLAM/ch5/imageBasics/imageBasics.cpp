#include <iostream>
#include <chrono>//用于算法计时
 // chrono是一个时间库, 源于boost，现在已经是C++标准。

//#include <opencv2/opencv.hpp>//这个头文件包含了 大部分头文件
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> // cv::cvtColor 函数


using namespace std;
using namespace cv;

//官方介绍文档
// https://docs.opencv.org/3.0-rc1/df/d65/tutorial_table_of_content_introduction.html
/*
 
    1 载入图像    Load an image (using                    cv::imread )     Mat img = imread(filename, 0);（读成灰度图）
    2 创建窗口    Create a named OpenCV window (using     cv::namedWindow )
    3 显示图像     Display an image in an OpenCV window (using    cv::imshow )
    4 格式转换     cv::cvtColor( image, gray_image, COLOR_BGR2GRAY );
    5 保存图像     cv:: imwrite( "..//Gray_Image.jpg", gray_image );
    6 拷贝图像     cv::Mat image_clone = image.clone();//复制数据到另一块内存空间
                           cv::Mat image_clone ;  image.copyTo(image_clone ); 也可复制
 
 */
/*
 //CMake file
 cmake_minimum_required(VERSION 2.8)//版本限制
project( DisplayImage )//工程名
find_package( OpenCV REQUIRED )//找到安装包位置
include_directories( ${OpenCV_INCLUDE_DIRS} )//添加头文件
add_executable( DisplayImage DisplayImage.cpp )//添加可执行文件
target_link_libraries( DisplayImage ${OpenCV_LIBS} )//添加动态链接库
 */

int main ( int argc, char** argv )
{
  
      if ( argc != 2 )
    {
        //printf("请指定图像的文件名路径\n");
       cerr<<"请指定图像的文件名路径."<<endl;//输出到错误流
        return -1;
    }
    
   string imageName("../ubuntu.png"); //默认文件名
    // 读取argv[1]指定的图像
    cv::Mat image;
    if( argc > 1)   imageName = argv[1];// create a Mat object
   // image = cv::imread ( imageName.c_str(), IMREAD_COLOR ); //cv::imread函数读取指定路径下的图像 ../ubuntu.png
    image = cv::imread ( imageName.c_str());
    /*
     
    备注：
    如果flag>0，返回一个三通道的彩色图像(强转)，
    flag=0返回一个灰度图像(强转)，
    flag<0,返回包含Alpha通道的原始图像(不修改通道数)
    用法举例：
        Mat image1 = imread("try,jpg", 2 | 4);//载入无损的原图像  
        Mat image2 = imread("try,jpg", 0);    //载入灰度图像  
        Mat image3 = imread("try,jpg", 199);  //载入三通道彩色图像  
     */
    // 判断图像文件是否正确读取
    if ( image.data == nullptr ) // 返回空指针   数据不存在,可能是文件不存在
      //  if ( !image.data )
      // if( image.empty() ) 
    {
        cerr<<"文件"<<argv[1]<<"不存在."<<endl;//输出到错误流
        return 0;
    }
    
    // 文件顺利读取, 首先输出一些基本信息 行数为高　　列数为宽
    //image.cols列数(宽)  image.rows行数(高)  image.channels()图像通道数
    cout<<"图像宽为"<<image.cols<<",高为"<<image.rows<<",通道数为"<<image.channels()<<endl;
    
    // namedWindow("显示窗口名子", WINDOW_AUTOSIZE ); 也可以先指定窗口名字
    cv::imshow ( argv[1], image );      // 前一个参数为 窗口名字  用cv::imshow显示图像
    
   
    cv::waitKey ( 0 );                  // 暂停程序,等待一个按键输入　随机按键
 
    // 判断image的类型　　CV_8UC1   1通道８位无符号 灰度图　　CV_8UC３   ３通道８位无符号  彩色图
    if ( image.type() != CV_8UC1 && image.type() != CV_8UC3 )
    {
        // 图像类型不符合要求
        cout<<"请输入一张彩色图或灰度图."<<endl;
        return 0;
    }
   
   //彩色图转灰度图
   cv::Mat gray_image;
   cv::cvtColor( image, gray_image, cv::COLOR_BGR2GRAY ); // COLOR_GRAY2BGR
   // 写图像文件
  cv:: imwrite( "..//Gray_Image.jpg", gray_image );
   cv::imshow ( "Gray_Image" , gray_image);//显示原来的图像   被修改了
   
   
   /*
    cv::Mat::isContinuous() 和 Mat::ptr<uchar>(i)  结合  可以提速
    这个跟计算机组成有关，关于ip寄存器的。其中一个结论是下面的代码，前面比后面快：
    */
   
   //     遍历图像, 请注意以下遍历方式亦可使用于随机像素访问
   // 1 指针直接访问 对一个对象Mat，通过调用函数  Mat::ptr<uchar>(i)  来得到第i行的指针地址
    // 使用 std::chrono 来给算法计时
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//当前时间点  steady_clock表示系统时钟 不受影响
    for ( size_t y=0; y<image.rows; y++ )//行坐标为y 
    {
        for ( size_t x=0; x<image.cols; x++ )//l列坐标为x
        {
            // 访问位于 x,y 处的像素
            // 用cv::Mat::ptr获得图像的行指针
            unsigned char* row_ptr = image.ptr<unsigned char> ( y );  // row_ptr是第y行的头指针
            unsigned char* data_ptr = &row_ptr[ x*image.channels() ]; // & 取地址   data_ptr 指向待访问的像素数据  每个通道 对应点的像素值
            // 输出该像素的每个通道,如果是灰度图就只有一个通道
            for ( int c = 0; c != image.channels(); c++ )
            {
                unsigned char data = data_ptr[c]; // data为I(x,y)第c个通道的值
            }
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//算法运行后 此刻时间点
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );//时间差
    cout<<"指针遍历图像用时："<<time_used.count()<<" 秒。"<<endl;  //最快

// 2、迭代器访问
    /*
对一个Mat，
创建一个Mat::Iterator对象it和itend，
通过it=Mat::begin()来的到迭代首地址，
itend=Mat::end()来得到尾地址，
it != itend来判断是否到尾，
it++来得到下一个像素指向，
(*it)来得到实际像素  *取地址内的值
    */
  Mat_<Vec3b>::iterator it =  image.begin<Vec3b>();  //迭代 首地址
  Mat_<Vec3b>::iterator itend =  image.end<Vec3b>(); //迭代 尾地址
    chrono::steady_clock::time_point t11 = chrono::steady_clock::now();//当前时间点  steady_clock表示系统时钟 不受影响
   while (it != itend)  
    {  
            // 输出该像素的每个通道,如果是灰度图就只有一个通道
            for ( int c = 0; c != image.channels(); c++ )
            {
                unsigned char data = (*it)[c]; //  (*it)[0]代表当前像素单位的B位，(*it)[1]代表当前像素单位的G位，(*it)[2]代表当前像素单位的R位
            }
            it++;  
    }

    chrono::steady_clock::time_point t22 = chrono::steady_clock::now();//算法运行后 此刻时间点
    chrono::duration<double> time_used1 = chrono::duration_cast<chrono::duration<double>>( t22-t11 );//时间差
    cout<<"迭代器遍历图像用时："<<time_used1.count()<<" 秒。"<<endl;//较快

    
    
  /*  
 3、动态访问       用类自带的方法image.at<Vec3b>(i,j)[c] 方便，但效率不高
这种方法是最慢的
对一个mat，可以直接用at函数来得到像素，Mat::at<Vec3b>(i,j)为一个像素点
*/
int colNum= image.cols;    //  列  宽 
int rowN=image.rows;       //行    高
    chrono::steady_clock::time_point t111 = chrono::steady_clock::now();//当前时间点  steady_clock表示系统时钟 不受影响
for (int i = 0; i < rowN; i++)  
    {  
        for (int j = 0; j < colNum; j++)        //这里colNum要注意，下面说明  
        { 
            // 输出该像素的每个通道,如果是灰度图就只有一个通道
            for ( int c = 0; c != image.channels(); c++ )
            {
                unsigned char data =image.at<Vec3b>(i,j)[c]; //  (*it)[0]代表当前像素单位的B位，(*it)[1]代表当前像素单位的G位，(*it)[2]代表当前像素单位的R位
            }
        }  
    }  
    chrono::steady_clock::time_point t222 = chrono::steady_clock::now();//算法运行后 此刻时间点
    chrono::duration<double> time_used2 = chrono::duration_cast<chrono::duration<double>>( t222-t111 );//时间差
    cout<<"动态访问遍历图像用时："<<time_used2.count()<<" 秒。"<<endl;//最慢。

    
    
    // 关于 cv::Mat 的拷贝
    // 直接赋值并不会拷贝数据
    cv::Mat image_another = image;//只是赋值的一个指针
    // 修改 image_another 会导致 image 发生变化
    image_another ( cv::Rect ( 0,0,100,100 ) ).setTo ( 0 ); // 将左上角100*100的块置零　变白色
    cv::imshow ( argv[1], image );//显示原来的图像   被修改了
    cv::waitKey ( 0 );
    
    // 使用clone函数来拷贝数据
    cv::Mat image_clone = image.clone();//复制数据到另一块内存空间
    // cv::Mat image_clone ;  image.copyTo(image_clone ); 也可复制
    
    image_clone ( cv::Rect ( 0,0,100,100 ) ).setTo ( 255 );//变黑色
    cv::imshow ( "image", image );//原图像为改变
    cv::imshow ( "image_clone", image_clone );//新图像改变
    cv::waitKey ( 0 );

    // 对于图像还有很多基本的操作,如剪切,旋转,缩放等,限于篇幅就不一一介绍了,请参看OpenCV官方文档查询每个函数的调用方法.
    cv::destroyAllWindows();
    return 0;
}
