/*
载入、修改、保存图片
*/

#include <opencv2/opencv.hpp>
#include <string>
using std::string;
using namespace cv;
int main( int argc, char** argv )
{
	string imageName("../data/77.jpeg"); // 图片文件名路径（默认值）
	    if( argc > 1)
	    {
		imageName = argv[1];//如果传递了文件 就更新
	    }
	 Mat image;
	 image = imread( imageName.c_str(), IMREAD_COLOR); // 按源图片颜色显示
	    // IMREAD_GRAYSCALE  灰度图格式读取 
	 if(!image.data )//image.empty()
	 {
	   std::cout <<  "打不开图片 image" << std::endl ;
	   return -1;
	 }
	//转换成灰度图
	 Mat gray_image;
	 cvtColor( image, gray_image, COLOR_BGR2GRAY );
// COLOR_BGR2BGRA  添加 透明通道   COLOR_BGR2RGB   BGR 顺序到 RGB顺序  COLOR_GRAY2BGR 灰度图到彩色图
// https://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html#gga4e0972be5de079fed4e3a10e24ef5ef0a353a4b8db9040165db4dacb5bcefb6ea
	//存储图片
	 imwrite( "../data/Gray_77.jpeg", gray_image );
	//窗口显示
	 namedWindow( imageName, WINDOW_AUTOSIZE );//原图
	 namedWindow( "Gray image", WINDOW_AUTOSIZE );//灰度图
	 imshow( imageName, image );
	 imshow( "Gray image", gray_image );

	 waitKey(0);// 等待窗口中的 按键响应 程序结束
	 return 0;
}

