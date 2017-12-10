/* This is sample from the OpenCV book. The copyright notice is below
 * 双目相机 矫正 
 * 用法 ./stereo_calib -w=6 -h=8  -s=2.43 stereo_calib.xml   我的
 * ./stereo_calib -w=9 -h=6 ../date/std/stereo_calib.xml 标准图像
 *  实际的正方格尺寸在程序中指定 const float squareSize = 2.43f;    2.43cm
 * https://www.cnblogs.com/polly333/p/5013505.html 原理分析
 * http://blog.csdn.net/zc850463390zc/article/details/48975263
   ************************************************** */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>//容器
#include <string>//字符串
#include <algorithm>//算法
#include <iostream>//输入输出流
#include <iterator>//迭代器
#include <stdio.h>//标志io
#include <stdlib.h>//标志库
#include <ctype.h>//c标志函数库

using namespace cv;
using namespace std;

static int print_help()
{
    cout <<
            " Given a list of chessboard images, the number of corners (nx, ny)\n"
            " on the chessboards, and a flag: useCalibrated for \n"
            "   calibrated (0) or\n"
            "   uncalibrated \n"
            "     (1: use cvStereoCalibrate(), 2: compute fundamental\n"
            "         matrix separately) stereo. \n"
            " Calibrate the cameras and display the\n"
            " rectified results along with the computed disparity images.   \n" << endl;
    cout << "Usage:\n ./stereo_calib -w=<board_width default=9> -h=<board_height default=6> <image list XML/YML file default=../data/stereo_calib.xml>\n" << endl;
    return 0;
}


static void
StereoCalib(const vector<string>& imagelist, Size boardSize, float squareSize_, bool  displayCorners = false, bool useCalibrated=true, bool showRectified=true)
{
    if( imagelist.size() % 2 != 0 )//成对的图像
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    const int maxScale = 2;
    const float squareSize = squareSize_;  // Set this to your actual square size 我的 实际的正方格尺寸  2.43cm 原来作者的 1.0f
    // ARRAY AND VECTOR STORAGE:
    
 //创建图像坐标和世界坐标系坐标矩阵
    vector<vector<Point2f> > imagePoints[2];//图像点(存储角点) 
    vector<vector<Point3f> > objectPoints;//物体三维坐标点
    Size imageSize;

    int i, j, k, nimages = (int)imagelist.size()/2;//左右两个相机图片
//确定左右视图矩阵的数量，比如10副图，左右矩阵分别为5个
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImageList;

    //  文件列表 需要交替 "left01.jpg" "right01.jpg"...
    for( i = j = 0; i < nimages; i++ )//5对图像
    {
        for( k = 0; k < 2; k++ )//左右两个相机图片 k=0,1
        {
	   //逐个读取图片
            const string& filename = imagelist[i*2+k];
	    Mat img_src,img;
            img_src = imread(filename, 1);//图像数据
	   cvtColor(img_src, img, COLOR_BGR2GRAY);//转换到 灰度图
            if(img.empty())//图像打不开
                break;
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )// 图片需要保持一样的大小
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
	    //设置图像矩阵的引用(指针)，此时指向左右视图的矩阵首地址
            vector<Point2f>& corners = imagePoints[k][j];//指向 每个图片的角点容器的首地址
            for( int scale = 1; scale <= maxScale; scale++ )
            {
                Mat timg;
		//图像是8bit的灰度或彩色图像
                if( scale == 1 )
                    timg = img;
                else
		  //如果为多通道图像
                    resize(img, timg, Size(), scale, scale);//转换成 8bit的灰度或彩色图像
		    //参数需为 8bit的灰度或彩色图像
                found = findChessboardCorners(timg, boardSize, corners,//得到棋盘内角点坐标  存入 imagePoints[k][j] 中
                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
                if( found )
                {
		  //如果为多通道图像
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }
            //显示角点
            if( displayCorners )
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);//转换成灰度图
                drawChessboardCorners(cimg, boardSize, corners, found);//显示
                double sf = 640./MAX(img.rows, img.cols);//尺度因子
                resize(cimg, cimg1, Size(), sf, sf);//变换到合适大小
                imshow("corners", cimg1);
                char c = (char)waitKey(500);//等待500ms
                if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC (27) to quit ESC键可退出
                    exit(-1);
            }
            else
                putchar('.');
            if( !found )
                break;
	    // 亚像素级优化
            cornerSubPix(img, corners, Size(11,11), Size(-1,-1),
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                      30, 0.01));
        }
        if( k == 2 )//上面的for循环结束后 k=2
        {
            goodImageList.push_back(imagelist[i*2]);
            goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if( nimages < 2 )
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages);//左相机 角点位置
    imagePoints[1].resize(nimages);//右相机 角点位置  
    // 角点实际 位置 按照 squareSize 得出
    objectPoints.resize(nimages);

    for( i = 0; i < nimages; i++ )//5对图
    {
        for( j = 0; j < boardSize.height; j++ )//每一行
            for( k = 0; k < boardSize.width; k++ )//每一列
                objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));//直接转为float类型，坐标为行、列
    }

    cout << "Running stereo calibration ...\n";
   //创建内参矩阵
    Mat cameraMatrix[2], distCoeffs[2];//左右两个相机 的内参数矩阵和 畸变参数
    cameraMatrix[0] = initCameraMatrix2D(objectPoints,imagePoints[0],imageSize,0);//初始化内参数矩阵
    cameraMatrix[1] = initCameraMatrix2D(objectPoints,imagePoints[1],imageSize,0);
    Mat R, T, E, F; //R 旋转矢量 T平移矢量 E本征矩阵 F基础矩阵  

    // 求解获取双标定的参数
    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],//真实点坐标  左右两个相机点坐标
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F,
                    CALIB_FIX_ASPECT_RATIO +
                    CALIB_ZERO_TANGENT_DIST +
                    CALIB_USE_INTRINSIC_GUESS +
                    CALIB_SAME_FOCAL_LENGTH +
                    CALIB_RATIONAL_MODEL +
                    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    cout << "done with RMS error=" << rms << endl;

// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    //计算标定误差
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];//极线
    for( i = 0; i < nimages; i++ )
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for( k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);//未去畸变
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for( j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average epipolar err = " <<  err/npoints << endl;

    // 内参数 save intrinsic parameters
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    //外参数
// R--右相机相对左相机的旋转矩阵
// T--右相机相对左相机的平移矩阵
// R1,R2--左右相机校准变换（旋转）矩阵  3×3
// P1,P2--左右相机在校准后坐标系中的投影矩阵 3×4
// Q--视差-深度映射矩阵，我利用它来计算单个目标点的三维坐标
    Mat R1, R2, P1, P2, Q;//由stereoRectify()求得
    Rect validRoi[2];//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  

    // 校准双目图像   摆正两幅图像
    // https://en.wikipedia.org/wiki/Image_rectification
    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
 /* 
    立体校正的时候需要两幅图像共面并且行对准 以使得立体匹配更加的可靠 
    使得两幅图像共面的方法就是把两个摄像头的图像投影到一个公共成像面上，这样每幅图像从本图像平面投影到公共图像平面都需要一个旋转矩阵R 
    stereoRectify 这个函数计算的就是从图像平面投影都公共成像平面的旋转矩阵Rl,Rr。 Rl,Rr即为左右相机平面行对准的校正旋转矩阵。 
    左相机经过Rl旋转，右相机经过Rr旋转之后，两幅图像就已经共面并且行对准了。 
    其中Pl,Pr为两个相机的投影矩阵，其作用是将3D点的坐标转换到图像的2D点的坐标:P*[X Y Z 1]' =[x y w]  
    Q矩阵为重投影矩阵，即矩阵Q可以把2维平面(图像平面)上的点投影到3维空间的点:Q*[x y d 1] = [X Y Z W]。其中d为左右两幅图像的时差 
    */ 
    fs.open("extrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    // 辨认 左右结构的相机  或者 上下结构的相机
    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

// COMPUTE AND DISPLAY RECTIFICATION
    if( !showRectified )
        return;

    Mat rmap[2][2]; //映射表 
// IF BY CALIBRATED (BOUGUET'S METHOD)
    if( useCalibrated )
    {
        // we already computed everything
    }
// OR ELSE HARTLEY'S METHOD
    else
 // use intrinsic parameters of each camera, but
 // compute the rectification transformation directly
 // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for( k = 0; k < 2; k++ )
        {
            for( i = 0; i < nimages; i++ )
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }
    /* 
    根据stereoRectify 计算出来的R 和 P 来计算图像的映射表 mapx,mapy 
    mapx,mapy这两个映射表接下来可以给remap()函数调用，来校正图像，使得两幅图像共面并且行对准 
    ininUndistortRectifyMap()的参数newCameraMatrix就是校正后的摄像机矩阵。在openCV里面，校正后的计算机矩阵Mrect是跟投影矩阵P一起返回的。 
    所以我们在这里传入投影矩阵P，此函数可以从投影矩阵P中读出校正后的摄像机矩阵 
    */ 
    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    /* 
        把校正结果显示出来 
        把左右两幅图像显示到同一个画面上 
        这里只显示了最后一副图像的校正结果。并没有把所有的图像都显示出来 
    */  
    Mat canvas;
    double sf;
    int w, h;
    if( !isVerticalStereo )
    {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else
    {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }

    for( i = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            Mat img = imread(goodImageList[i*2+k], 0), rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);//经过remap之后，左右相机的图像已经共面并且行对准了 
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));//得到画布的一部分 
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            if( useCalibrated )
            {
                Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf)); //获得被截取的区域
                rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);////画上一个矩形
            }
        }

        //画上对应的线条
        if( !isVerticalStereo )
            for( j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);//画平行线
        else
            for( j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
        char c = (char)waitKey();
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
    }
}


static bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}

int main(int argc, char** argv)
{
    Size boardSize;
    string imagelistfn;
    bool showRectified;
     float squareSize;// 格子大小
    cv::CommandLineParser parser(argc, argv, "{w|9|}{h|6|}{s|1|}{nr||}{help||}{@input|../data/std/stereo_calib.xml|}");
    if (parser.has("help"))
        return print_help();
    showRectified = !parser.has("nr");
    imagelistfn = parser.get<string>("@input");
    boardSize.width = parser.get<int>("w");
    boardSize.height = parser.get<int>("h");
    squareSize = parser.get<float>("s");//正方形棋盘格子 边长  
    if ( squareSize <= 0 )//标定板 格子尺寸参数错误
        return fprintf( stderr, "Invalid board square width\n" ), -1;
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    vector<string> imagelist;
    bool ok = readStringList(imagelistfn, imagelist);
    if(!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        return print_help();
    }

    StereoCalib(imagelist, boardSize, squareSize, true, true, showRectified);//
    return 0;
}
