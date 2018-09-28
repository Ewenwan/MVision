//我的 pc 摄像头 ./Monocular_Calibr  -w=6 -h=8 -s=2.43 -o=camera.yml -op -oe
//./Monocular_Calibr -w=8 -h=10 -s=200 -o=camera.yml -op -oe
// 离线校正
//./Monocular_Calibr -w=8 -h=10 -s=200 -o=wcamera.yml -op -oe wimagelist.yaml
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>

using namespace cv;
using namespace std;

// 用法信息   字符串字面值 常量 const char * 
const char * usage =
" \nexample command line for calibration from a live feed.\n" //在线矫正
"   calibration  -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe\n"// w宽度 一行上的内角点数  h一列上的内角点数 s每个格子的长度(cm) 输出标定文件
" \n"
" example command line for calibration from a list of stored images:\n"//从图片文件夹中标定
"   imagelist_creator image_list.xml *.png\n"//先生成 图片文件路径 yaml文件

"   calibration -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe image_list.xml\n"//从图片文件夹中标定
" where image_list.xml is the standard OpenCV XML/YAML\n"
" use imagelist_creator to create the xml or yaml list\n"
" file consisting of the list of strings, e.g.:\n"
" \n"
"<?xml version=\"1.0\"?>\n"
"<opencv_storage>\n"
"<images>\n"
"view000.png\n"
"view001.png\n"
"<!-- view002.png -->\n"
"view003.png\n"
"view010.png\n"
"one_extra_view.jpg\n"
"</images>\n"
"</opencv_storage>\n";

// 在线矫正帮助   字符串字面值 常量 const char * 
const char* liveCaptureHelp =
    "When the live video from camera is used as input, the following hot-keys may be used:\n"
        "  <ESC>, 'q' - quit the program\n"//暂停
        "  'g' - start capturing images\n"   //开始捕获突破
        "  'u' - switch undistortion on/off\n";//图像去畸变 undistortion

//帮助信息 函数
static void help()
{
    printf( "This is a camera calibration sample.\n"
        "Usage: calibration\n"
        "     -w=<board_width>         # the number of inner corners per one of board dimension\n"//一行上的角点数
        "     -h=<board_height>        # the number of inner corners per another board dimension\n"//一列上的角点数
        "     [-pt=<pattern>]          # the type of pattern: chessboard or circles' grid\n"//标定板类型 棋盘格子  还是 圆形格子
        "     [-n=<number_of_frames>]  # the number of frames to use for calibration\n"
        "                              # (if not specified, it will be set to the number\n"
        "                              #  of board views actually available)\n"
        "     [-d=<delay>]     # a minimum delay in ms between subsequent attempts to capture a next view\n"//捕获 时间间隔
        "                              # (used only for video capturing)\n"
        "     [-s=<squareSize>]       # square size in some user-defined units (1 by default)\n"//正方形格子大小
        "     [-o=<out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"//输出标定文件名
        "     [-op]                    # write detected feature points\n"//带有检测到的特征点
        "     [-oe]                    # write extrinsic parameters\n"//带有外参数  R+T
        "     [-zt]                    # assume zero tangential distortion\n"
        "     [-a=<aspectRatio>]      # fix aspect ratio (fx/fy)\n"//:固定fx/fy的比值,只将fy作为可变量,进行优化计算
        "     [-p]                     # fix the principal point at the center\n"
        "     [-v]                     # flip the captured images around the horizontal axis\n"//在水平轴周围翻转拍摄的图像
        "     [-V]                     # use a video file, and not an image list, uses\n"//使用 视频文件
        "                              # [input_data] string for the video file name\n"
        "     [-su]                    # show undistorted images after calibration\n"
        "     [input_data]             # input data, one of the following:\n"
        "                              #  - text file with a list of the images of the board\n"
        "                              #    the text file can be generated with imagelist_creator\n"
        "                              #  - name of video file with a video of the board\n"
        "                              # if input_data not specified, a live view from the camera is used\n"
        "\n" );
    printf("\n%s",usage);//用法信息
    printf( "\n%s", liveCaptureHelp );//在线矫正 帮助信息
}

//枚举  符号常量
enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };//标定板类型  棋盘格子     对称圆形标志检测  非对称圆形标定物检测

// 计算重投影误差
static double computeReprojectionErrors(
        const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs,
        vector<float>& perViewErrors )
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);//重投影
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);//重投影误差
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }

    return std::sqrt(totalErr/totalPoints);
}

//获取棋盘格 内角点  (按给定参数) 标准位置 groundtrouth
static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
    corners.resize(0);

    switch(patternType)
    {
      case CHESSBOARD:
      case CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float(j*squareSize),
                                          float(i*squareSize), 0));//按给定参数 得到标准 焦点位置
        break;

      case ASYMMETRIC_CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float((2*j + i % 2)*squareSize),
                                          float(i*squareSize), 0));
        break;

      default:
        CV_Error(Error::StsBadArg, "Unknown pattern type\n");
    }
}

// 进行矫正
static bool runCalibration( vector<vector<Point2f> > imagePoints,
                    Size imageSize, Size boardSize, Pattern patternType,
                    float squareSize, float aspectRatio,
                    int flags, Mat& cameraMatrix, Mat& distCoeffs,
                    vector<Mat>& rvecs, vector<Mat>& tvecs,
                    vector<float>& reprojErrs,
                    double& totalAvgErr)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);//[fx,0,ux; 0,fy,uy; 0,0,1] 内参数
    if( flags & CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = aspectRatio;

    distCoeffs = Mat::zeros(8, 1, CV_64F);//畸变参数  k1,k2,k3,k4,k5,k6为径向畸变，p1,p2为切向畸变
 
    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);//获取真实点位置

    objectPoints.resize(imagePoints.size(),objectPoints[0]);
    
// objectPoints真实点位置  imagePoints从照片中检测到的点坐标
    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                    distCoeffs, rvecs, tvecs, flags|CALIB_FIX_K4|CALIB_FIX_K5);
                    ///*|CALIB_FIX_K3*/|CALIB_FIX_K4|CALIB_FIX_K5);
    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}


//
static void saveCameraParams( const string& filename,
                       Size imageSize, Size boardSize,
                       float squareSize, float aspectRatio, int flags,
                       const Mat& cameraMatrix, const Mat& distCoeffs,
                       const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                       const vector<float>& reprojErrs,
                       const vector<vector<Point2f> >& imagePoints,
                       double totalAvgErr )
{
    FileStorage fs( filename, FileStorage::WRITE );

    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;// 矫正时间  calibration_time: "17/12/13 13:08:39"

    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());//帧数
    fs << "image_width" << imageSize.width;//图像宽度
    fs << "image_height" << imageSize.height;//图像高度
    fs << "board_width" << boardSize.width;// 标定板角点 数 一行
    fs << "board_height" << boardSize.height;// 标定板角点 数 一列
    fs << "square_size" << squareSize;// 标定板 单个格子尺寸

    if( flags & CALIB_FIX_ASPECT_RATIO )
        fs << "aspectRatio" << aspectRatio;// fx/fx固定比例

    if( flags != 0 )
    {
        sprintf( buf, "flags: %s%s%s%s",
            flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
            flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
            flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
            flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
        //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;//标志

    fs << "camera_matrix" << cameraMatrix;//内参数矩阵
    fs << "distortion_coefficients" << distCoeffs;//畸变矩阵

    fs << "avg_reprojection_error" << totalAvgErr;// 平均重投影误差
    if( !reprojErrs.empty() )
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(i, i+1), Range(0,3));
            Mat t = bigmat(Range(i, i+1), Range(3,6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }

    if( !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }
}

//
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

// 进行矫正 并保存 矫正的参数文件
static bool runAndSave(const string& outputFilename,
                const vector<vector<Point2f> >& imagePoints,
                Size imageSize, Size boardSize, Pattern patternType, float squareSize,
                float aspectRatio, int flags, Mat& cameraMatrix,
                Mat& distCoeffs, bool writeExtrinsics, bool writePoints )
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
                   aspectRatio, flags, cameraMatrix, distCoeffs,
                   rvecs, tvecs, reprojErrs, totalAvgErr);
    printf("%s. avg reprojection error = %.2f\n",
           ok ? "Calibration succeeded" : "Calibration failed",
           totalAvgErr);

    if( ok )
        saveCameraParams( outputFilename, imageSize,
                         boardSize, squareSize, aspectRatio,
                         flags, cameraMatrix, distCoeffs,
                         writeExtrinsics ? rvecs : vector<Mat>(),
                         writeExtrinsics ? tvecs : vector<Mat>(),
                         writeExtrinsics ? reprojErrs : vector<float>(),
                         writePoints ? imagePoints : vector<vector<Point2f> >(),
                         totalAvgErr );
    return ok;
}


// 主函数
int main( int argc, char** argv )
{
  //***********************
  //参数定义
    Size boardSize, imageSize;//标定板 尺寸(角点个数)  图像尺寸
    float squareSize, aspectRatio;// 格子大小  aspectRatio :固定fx/fy的比值,只将fy作为可变量,进行优化计算
    Mat cameraMatrix, distCoeffs;// 相机内参数矩阵
    string outputFilename;//输出文件 名
    string inputFilename = "";//输入文件名

    int i, nframes;
    bool writeExtrinsics, writePoints;
    bool undistortImage = false;
    int flags = 0;
    VideoCapture capture;//捕获相机对象
    bool flipVertical;
    bool showUndistorted;// 显示 未去 畸变
    bool videofile;
    int delay;
    clock_t prevTimestamp = 0;
    int mode = DETECTION;
    int cameraId = 0;
    vector<vector<Point2f> > imagePoints;//从图像中找到的 角点 坐标 容器
    vector<string> imageList;//图像 列表容器
    Pattern pattern = CHESSBOARD;//默认为棋盘格
    //********************************
    // 解析命令行参数  -w |默认参数|
    cv::CommandLineParser parser(argc, argv,
        "{help ||}{w||}{h||}{pt|chessboard|}{n|10|}{d|1000|}{s|1|}{o|out_camera_data.yml|}"
        "{op||}{oe||}{zt||}{a|1|}{p||}{v||}{V||}{su||}"
        "{@input_data|0|}");
    //******************************
    // 帮助信息
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    //******************************
    //标定板 尺寸(角点个数) 
    boardSize.width = parser.get<int>( "w" );
    boardSize.height = parser.get<int>( "h" );
    //指定 标定板格式
    if ( parser.has("pt") )
    {
        string val = parser.get<string>("pt");
        if( val == "circles" )
            pattern = CIRCLES_GRID;//对称圆形
        else if( val == "acircles" )
            pattern = ASYMMETRIC_CIRCLES_GRID;//非对称圆形
        else if( val == "chessboard" )//棋盘格
            pattern = CHESSBOARD;
        else
            return fprintf( stderr, "Invalid pattern type: must be chessboard or circles\n" ), -1;//其他模式非法 输出到标志错误输出流 fprintf(stderr,...)
    }
    //******************************
    // 获取命令行指定的参数
    squareSize = parser.get<float>("s");//正方形棋盘格子 边长  
    nframes = parser.get<int>("n");// 线性矫正模型  每隔几帧 图像 采样一次
    aspectRatio = parser.get<float>("a");//aspectRatio :固定fx/fy的比值,只将fy作为可变量,进行优化计算
    delay = parser.get<int>("d");//捕获 时间间隔
    writePoints = parser.has("op");//带有检测到的特征点
    writeExtrinsics = parser.has("oe");//带有外参数 R+T
    if (parser.has("a"))
        flags |= CALIB_FIX_ASPECT_RATIO;//固定fx/fy的比值
    if ( parser.has("zt") )
        flags |= CALIB_ZERO_TANGENT_DIST;
    if ( parser.has("p") )
        flags |= CALIB_FIX_PRINCIPAL_POINT;
    flipVertical = parser.has("v");//在水平轴周围翻转拍摄的图像
    videofile = parser.has("V");//从视频文件流中 标定
    if ( parser.has("o") )//输出文件名
        outputFilename = parser.get<string>("o");
    showUndistorted = parser.has("su");//显示  未 去畸变的图片
    if ( isdigit(parser.get<string>("@input_data")[0]) )//是数字的话  为摄像头ID
        cameraId = parser.get<int>("@input_data");
    else
        inputFilename = parser.get<string>("@input_data");//否则为 文件名  yaml 或者视频文件
	
//************************×××××××××
//检查参数是否合法	
  // 参数有问题
    if (!parser.check())
    {
        help();//显示 帮助信息
        parser.printErrors();
        return -1;//退出
    }
    if ( squareSize <= 0 )//标定板 格子尺寸参数错误
        return fprintf( stderr, "Invalid board square width\n" ), -1;
    if ( nframes <= 3 )
        return printf("Invalid number of images\n" ), -1;
    if ( aspectRatio <= 0 )
        return printf( "Invalid aspect ratio\n" ), -1;
    if ( delay <= 0 )
        return printf( "Invalid delay\n" ), -1;
    if ( boardSize.width <= 0 )//标定板内角点参数错误
        return fprintf( stderr, "Invalid board width\n" ), -1;
    if ( boardSize.height <= 0 )
        return fprintf( stderr, "Invalid board height\n" ), -1;
    //********************************************
    // 标定图片来源
    if( !inputFilename.empty() )//文件名 飞空
    {
        if( !videofile && readStringList(inputFilename, imageList) )//读取照片文件
            mode = CAPTURING;
        else
            capture.open(inputFilename);//打开视频文件
    }
    else
        capture.open(cameraId);//在线矫正 打开相机 

    if( !capture.isOpened() && imageList.empty() )//打开相机失败
        return fprintf( stderr, "Could not initialize video (%d) capture\n",cameraId ), -2;

    if( !imageList.empty() )//
        nframes = (int)imageList.size();//总图片数量

    if( capture.isOpened() )//在线 矫正 
        printf( "%s", liveCaptureHelp );

    // 窗口
    namedWindow( "Image View", 1 );
//WINDOW_NORMAL设置了这个值，用户便可以改变窗口的大小（没有限制）
//WINDOW_AUTOSIZE如果设置了这个值，窗口大小会自动调整以适应所显示的图像，并且不能手动改变窗口大小。
//WINDOW_OPENGL 如果设置了这个值的话，窗口创建的时候便会支持OpenGL。

    for(i = 0;;i++)
    {
        Mat view, viewGray;//原图  灰度图
        bool blink = false;
//从相机设备中获取图片
        if( capture.isOpened() )
        {
            Mat view0;
            capture >> view0;//捕获一张
            view0.copyTo(view);//复制到 view
        }
        else if( i < (int)imageList.size() )
            view = imread(imageList[i], 1);//读取一张

        if(view.empty())//已经读取完毕
        {
            if( imagePoints.size() > 0 )
                runAndSave(outputFilename, imagePoints, imageSize,
                           boardSize, pattern, squareSize, aspectRatio,
                           flags, cameraMatrix, distCoeffs,
                           writeExtrinsics, writePoints);
            break;
        }

        imageSize = view.size();//图像大小

        if( flipVertical )//水平轴上下旋转
            flip( view, view, 0 );

        vector<Point2f> pointbuf;//存储二维点 的容器
        cvtColor(view, viewGray, COLOR_BGR2GRAY);//转换到 灰度图
	
// 获取图片的角点
        bool found; 
        switch( pattern )//标定板格式
        {//找角点
            case CHESSBOARD://棋盘
                found = findChessboardCorners( view, boardSize, pointbuf,
                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
                break;
            case CIRCLES_GRID://对称圆形
                found = findCirclesGrid( view, boardSize, pointbuf );
                break;
            case ASYMMETRIC_CIRCLES_GRID://非对称圆形
                found = findCirclesGrid( view, boardSize, pointbuf, CALIB_CB_ASYMMETRIC_GRID );
                break;
            default:
                return fprintf( stderr, "Unknown pattern type\n" ), -1;
        }

       // 提高角点坐标精度 improve the found corners' coordinate accuracy
        if( pattern == CHESSBOARD && found) cornerSubPix( viewGray, pointbuf, Size(11,11),
            Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ));
      // 在获取图像模式下 将角点加入 容器
        if( mode == CAPTURING && found &&
           (!capture.isOpened() || clock() - prevTimestamp > delay*1e-3*CLOCKS_PER_SEC) )
        {
            imagePoints.push_back(pointbuf);
            prevTimestamp = clock();
            blink = capture.isOpened();
        }

        if(found)//显示带角点的图像
            drawChessboardCorners( view, boardSize, Mat(pointbuf), found );
       //显示文字
        string msg = mode == CAPTURING ? "100/100" :
            mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";//从相机设备中获取图片 模式下
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

        if( mode == CAPTURING )
        {
            if(undistortImage)
                msg = format( "%d/%d Undist", (int)imagePoints.size(), nframes );//未 去畸变
            else
                msg = format( "%d/%d", (int)imagePoints.size(), nframes );
        }

        putText( view, msg, textOrigin, 1, 1,
                 mode != CALIBRATED ? Scalar(0,0,255) : Scalar(0,255,0));

        if( blink )
            bitwise_not(view, view);

        if( mode == CALIBRATED && undistortImage )
        {
            Mat temp = view.clone();
            undistort(temp, view, cameraMatrix, distCoeffs);
        }

        imshow("Image View", view);
        int key = 0xff & waitKey(capture.isOpened() ? 50 : 500);

        if( (key & 255) == 27 )
            break;

        if( key == 'u' && mode == CALIBRATED )//去 畸变
            undistortImage = !undistortImage;

        if( capture.isOpened() && key == 'g' )//在线矫正
        {
            mode = CAPTURING;
            imagePoints.clear();
        }

        if( mode == CAPTURING && imagePoints.size() >= (unsigned)nframes )
        {
            if( runAndSave(outputFilename, imagePoints, imageSize,
                       boardSize, pattern, squareSize, aspectRatio,
                       flags, cameraMatrix, distCoeffs,
                       writeExtrinsics, writePoints))
                mode = CALIBRATED;
            else
                mode = DETECTION;
            if( !capture.isOpened() )
                break;
        }
    }

    if( !capture.isOpened() && showUndistorted )//显示未 去畸变
    {
        Mat view, rview, map1, map2;
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                                getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                                imageSize, CV_16SC2, map1, map2);

        for( i = 0; i < (int)imageList.size(); i++ )
        {
            view = imread(imageList[i], 1);
            if(view.empty())
                continue;
            //undistort( view, rview, cameraMatrix, distCoeffs, cameraMatrix );
            remap(view, rview, map1, map2, INTER_LINEAR);
            imshow("Image View", rview);
            int c = 0xff & waitKey();
            if( (c & 255) == 27 || c == 'q' || c == 'Q' )
                break;
        }
    }

    return 0;
}
