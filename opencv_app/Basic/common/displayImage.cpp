/*
显示图片

mkdir build
cd build
cmake ..
make 
*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
int main( int argc, char** argv )
{
    string imageName("../data/77.jpeg"); // 图片文件名路径（默认值）
    if( argc > 1)
    {
        imageName = argv[1];//如果传递了文件 就更新
    }
    Mat image;//图片矩阵   string转char*
    image = imread(imageName.c_str(), IMREAD_COLOR); // 按源图片颜色显示
    // IMREAD_GRAYSCALE  灰度图格式读取 
    if( image.empty() ) // 检查图片是否读取成功
    {
        cout <<  "打不开图片 image" << std::endl ;
        return -1;
    }
   namedWindow( "Display window", WINDOW_AUTOSIZE ); // 创建一个窗口来显示图片.
   // namedWindow( "Display window", WINDOW_NORMAL );
  //namedWindow( "Display window", WINDOW_OPENGL );//opengl支持 编译时需要选择 opengl支持
// WINDOW_NORMAL 正常模型  用户可更改窗口大小  WINDOW_FULLSCREEN  全屏

    imshow( "Display window", image );                // 在窗口中显示图片.
    waitKey(0); // 等待窗口中的 按键响应 程序结束
    return 0;
}
