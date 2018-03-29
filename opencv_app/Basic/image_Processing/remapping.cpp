/*
重映射是什么意思?
    把一个图像中一个位置的像素放置到另一个图片指定位置的过程.
    为了完成映射过程, 有必要获得一些插值为非整数像素坐标,
    因为源图像与目标图像的像素坐标不是一一对应的.

    我们通过重映射来表达每个像素的位置 (x,y) :
   goal(x,y) = f(s(s,y))

*/ 

 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"
 #include <iostream>
 #include <stdio.h>

 using namespace cv;
 using namespace std;
 /// 全局变量
 Mat src, dst;
 Mat map_x, map_y;
 char* remap_window = "Remap demo";
 int ind = 0;

 /// 函数声明 更新映射关系
 void update_map( void );

/// 主函数
 int main( int argc, char** argv )
 {
  string imageName("../../common/data/77.jpeg"); // 图片文件名路径（默认值）
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

  /// 创建几张映射矩阵
// map_x: x方向的映射参数. 它相当于方法 h(i,j) 的第一个参数
// map_y: y方向的映射参数. 注意 map_y 和 map_x 与 src 的大小一致。
  dst.create( src.size(), src.type() );
  map_x.create( src.size(), CV_32FC1 );
  map_y.create( src.size(), CV_32FC1 );

  /// 创建显示窗口
  namedWindow( remap_window, CV_WINDOW_AUTOSIZE );

  /// 循环
  while( true )
  {
    int c = waitKey( 1000 );//1s检测一次按键  按Esc键退出
    if( (char)c == 27 )
      { break; }

    /// 更新重映射图
    update_map();
// map_x: x方向的映射参数. 它相当于方法 h(i,j) 的第一个参数
// map_y: y方向的映射参数. 注意 map_y 和 map_x 与 src 的大小一致。
    remap( src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );

    /// 显示结果
    imshow( remap_window, dst );
  }
  return 0;
 }

// 更新函数 
 void update_map( void )
 {
   ind = ind%4;// 0 1 2 3

   for( int j = 0; j < src.rows; j++ )//每行
   { for( int i = 0; i < src.cols; i++ )//每列
       {
         switch( ind )
         {
           case 0:// 图像宽高缩小一半，并显示在中间:
             if( i > src.cols*0.25 && i < src.cols*0.75 && j > src.rows*0.25 && j < src.rows*0.75 )
               {
                 map_x.at<float>(j,i) = 2*( i - src.cols*0.25 ) + 0.5 ;//记录的是坐标
                 map_y.at<float>(j,i) = 2*( j - src.rows*0.25 ) + 0.5 ;
                }
             else
               { map_x.at<float>(j,i) = 0 ;
                 map_y.at<float>(j,i) = 0 ;
               }
                 break;
           case 1:// 图像上下颠倒
                 map_x.at<float>(j,i) = i ;//列不变
                 map_y.at<float>(j,i) = src.rows - j ;//行交换
                 break;
           case 2:// 图像左右颠倒
                 map_x.at<float>(j,i) = src.cols - i ;//列交换
                 map_y.at<float>(j,i) = j ;//行不变
                 break;
           case 3:// 上下颠倒 + 左右颠倒
                 map_x.at<float>(j,i) = src.cols - i ;
                 map_y.at<float>(j,i) = src.rows - j ;
                 break;
         } // end of switch
       }
    }
  ind++;
}


