/*
利用形态学操作提取水平线和垂直线
腐蚀膨胀来提取线特征

*/
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
  string imageName("../../common/data/notes.png"); // 图片文件名路径（默认值）
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

    // 显示图像
    imshow("src", src);

    // 得到灰度图
    Mat gray;
    if (src.channels() == 3)//如果原图是彩色图
    {
        cvtColor(src, gray, CV_BGR2GRAY);//转换到灰度图
    }
    else
    {
        gray = src;
    }

    // 显示灰度图像
    imshow("gray", gray);

    // 灰度图二值化 ~ symbol
    Mat bw;
	       // 原图取反   输出  最大  自适应方法阈值           阈值类型  块大小
    adaptiveThreshold(~gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
    // THRESH_BINARY     大于阈值的都变为 255最大值 其余变为 0
    // THRESH_BINARY_INV 小于阈值的都变为 255最大值 其余变为 0 

    // 显示二值图
    imshow("binary", bw);
    // 创建水平线和垂直线图像
    Mat horizontal = bw.clone();
    Mat vertical = bw.clone();

//========水平线提取 参考列数=====================
    int horizontalsize = horizontal.cols / 30;

    // 水平线 提取框  核子 窗口大小
    Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize,1));

    // 腐蚀+膨胀 =  开运算 (Opening)  去除 小型 白洞 保留水平白线
    erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    // 显示水平线
    imshow("horizontal", horizontal);


//========垂直线提取==========================
    int verticalsize = vertical.rows / 30;//
    // 核子 窗口大小
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size( 1,verticalsize));
    // 腐蚀+膨胀 =  开运算 (Opening)
    erode(vertical, vertical, verticalStructure, Point(-1, -1));
    dilate(vertical, vertical, verticalStructure, Point(-1, -1));
    // 显示垂直线
    imshow("vertical", vertical);

// 垂直线图 反向二值化
    bitwise_not(vertical, vertical);
    imshow("vertical_bit", vertical);

// Extract edges and smooth image according to the logic
// 1. extract edges
// 2. dilate(edges)
// 3. src.copyTo(smooth)
// 4. blur smooth img
// 5. smooth.copyTo(src, edges)
    //Step 1 提取边缘 
    Mat edges;
    adaptiveThreshold(vertical, edges, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
    imshow("edges", edges);
    // Step 2 膨胀操作
    Mat kernel = Mat::ones(2, 2, CV_8UC1);//核大小
    dilate(edges, edges, kernel);
    imshow("dilate", edges);
    // Step 3 得到平滑图像
    Mat smooth;
    vertical.copyTo(smooth);
    // Step 4 平滑图像
    blur(smooth, smooth, Size(2, 2));
    // Step 5
    smooth.copyTo(vertical, edges);
    // Show final result
    imshow("smooth", vertical);
    waitKey(0);
    return 0;
}

