/*
图像阈值操作
最简单的图像分割的方法。

应用举例：从一副图像中利用阈值分割出我们需要的物体部分
（当然这里的物体可以是一部分或者整体）。
这样的图像分割方法是基于图像中物体与背景之间的灰度差异，而且此分割属于像素级的分割。

为了从一副图像中提取出我们需要的部分，应该用图像中的每一个
像素点的灰度值与选取的阈值进行比较，并作出相应的判断。
（注意：阈值的选取依赖于具体的问题。即：物体在不同的图像中有可能会有不同的灰度值。

一旦找到了需要分割的物体的像素点，我们可以对这些像素点设定一些特定的值来表示。
（例如：可以将该物体的像素点的灰度值设定为：‘0’（黑色）,
其他的像素点的灰度值为：‘255’（白色）；当然像素点的灰度值可以任意，
但最好设定的两种颜色对比度较强，方便观察结果）。

【1】阈值类型1：二值阈值化
     大于阈值的 设置为最大值 255  其余为0
     先要选定一个特定的阈值量，比如：125，
    这样，新的阈值产生规则可以解释为大于125的像素点的灰度值设定为最大值(如8位灰度值最大为255)，
    灰度值小于125的像素点的灰度值设定为0。

【2】阈值类型2：反二进制阈值化
    小于阈值的 设置为最大值 255  其余为0


【3】阈值类型3：截断阈值化
    大于阈值的 设置为阈值  其余保持原来的值

【4】阈值类型4：阈值化为0
    大于阈值的 保持原来的值 其余设置为0

【5】阈值类型5：反阈值化为0 
    大于阈值的 设置为0  其余保持原来的值 


*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;

// 全局变量定义及赋值
int threshold_value = 0;
int threshold_type = 3;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst;
const char* window_name = "Threshold Demo";//窗口名
// 滑动条显示
const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";

const char* trackbar_value = "Value";

//阈值
void Threshold_Demo( int, void* );

int main( int argc, char** argv )
{
    string imageName("../../common/data/notes.png"); // 图片文件名路径（默认值）
    if( argc > 1)
    {
        imageName = argv[1];//如果传递了文件 就更新
    }

  src = imread( imageName );
  if( src.empty() )
    { 
        cout <<  "can't load image " << endl;
	return -1;  
    }
  //转成灰度图
  cvtColor( src, src_gray, COLOR_RGB2GRAY );
  //显示
  namedWindow( window_name, WINDOW_AUTOSIZE );

  // 阈值类型
  createTrackbar( trackbar_type,
                  window_name, &threshold_type,
                  max_type, Threshold_Demo );
  // 阈值大小
  createTrackbar( trackbar_value,
                  window_name, &threshold_value,
                  max_value, Threshold_Demo );
  // 初始化为
  Threshold_Demo( 0, 0 );
  //检测按键
  for(;;)
    {
      int c;
      c = waitKey( 20 );
      if( (char)c == 27 )//esc键退出
    { break; }
    }
}

void Threshold_Demo( int, void* )
{
  /* 0: Binary          二值
     1: Binary Inverted 二值反
     2: Threshold Truncated 截断
     3: Threshold to Zero   阈值化为0 大于阈值的 保持原来的值 其余设置为0
     4: Threshold to Zero Inverted  大于阈值的 设置为0  其余保持原来的值 
   */
  threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );
  imshow( window_name, dst );
}

