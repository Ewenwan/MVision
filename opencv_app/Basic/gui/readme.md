
# 为程序界面添加滑动条

# OpenCV的视频输入和相似度测量

    #include <iostream> // for standard I/O
    #include <string>   // for strings
    #include <iomanip>  // for controlling float print precision
    #include <sstream>  // string to number conversion

    #include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
    #include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
    #include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

    using namespace std;
    using namespace cv;


    // 需要先定义一个 VideoCapture 类的对象来打开和读取视频流。
    const string sourceReference = argv[1],sourceCompareWith = argv[2];

    VideoCapture captRefrnc(sourceReference);
    // 或者
    VideoCapture captUndTst;
    captUndTst.open(sourceCompareWith);

    如果使用整型数当参数的话，就可以将这个对象绑定到一个摄像机，
    将系统指派的ID号当作参数传入即可。例如你可以传入0来打开第一个摄像机，
    传入1打开第二个摄像机，以此类推。如果使用字符串当参数，

    VideoCapture cap(0);

    就会打开一个由这个字符串（文件名）指定的视频文件。


    ===================================================
    可以用 isOpened 函数来检查视频是否成功打开与否:

    if ( !captRefrnc.isOpened())
      {
      cout  << "Could not open reference " << sourceReference << endl;
      return -1;
      }

    ===============================================
    因为视频流是连续的，所以你需要在每次调用 read 函数后及时保存图像或者直接使用重载的>>操作符。
    Mat frameReference, frameUnderTest;
    captRefrnc >> frameReference;
    captUndTst.open(frameUnderTest);
    ===================
    如果视频帧无法捕获（例如当视频关闭或者完结的时候），上面的操作就会返回一个空的 Mat 对象。
    ============
    if( frameReference.empty()  || frameUnderTest.empty())
    {
     // 退出程序
    exit(0);
    }
    ==================================
    在下面的例子里我们会先获得视频的尺寸和帧数。

    Size refS = Size((int) captRefrnc.get(CV_CAP_PROP_FRAME_WIDTH),
                     (int) captRefrnc.get(CV_CAP_PROP_FRAME_HEIGHT)),

    cout << "参考帧的  宽度 =" << refS.width << "  高度 =" << refS.height << endl;

    =====================
    当你需要设置这些值的时候你可以调用 set 函数。
    captRefrnc.set(CV_CAP_PROP_POS_MSEC, 1.2);  // 跳转到视频1.2秒的位置
    captRefrnc.set(CV_CAP_PROP_POS_FRAMES, 10); // 跳转到视频的第10帧
    // 然后重新调用read来得到你刚刚设置的那一帧

    ===============================================
    图像比较 - PSNR 

    当我们想检查压缩视频带来的细微差异的时候，就需要构建一个能够逐帧比较差视频差异的系统。
    最常用的比较算法是PSNR( Peak signal-to-noise ratio)。
    这是个使用“局部均值误差”来判断差异的最简单的方法，
    假设有这两幅图像：I1和I2，它们的行列数分别是i，j，有c个通道。

    均值sqre误差 MSE   1/(i*j*c) * (I1 - I2)^2
    255 is max pix

    psnr = 10.0*log10((255*255)/mse);


    double getPSNR(const Mat& I1, const Mat& I2)
    {
     Mat s1;
     absdiff(I1, I2, s1);       // |I1 - I2|
     s1.convertTo(s1, CV_32F);  // 不能在8位矩阵上做平方运算
     s1 = s1.mul(s1);           // |I1 - I2|^2

     Scalar s = sum(s1);        // 叠加每个通道的元素 
     double sse = s.val[0] + s.val[1] + s.val[2]; // 叠加所有通道

     if( sse <= 1e-10) // 如果值太小就直接等于0
         return 0;
     else
     {
         double  mse = sse /(double)(I1.channels() * I1.total());
         double psnr = 10.0*log10((255*255)/mse);
         return psnr;
     }
    }

    在考察压缩后的视频时，psnr 这个值大约在30到50之间，
    数字越大则表明压缩质量越好。如果图像差异很明显，就可能会得到15甚至更低的值。
    PSNR算法简单，检查的速度也很快。但是其呈现的差异值有时候和人的主观感受不成比例。

    ================================
    所以有另外一种称作 结构相似性 structural similarity  的算法做出了这方面的改进。
    图像比较 - SSIM
    建议你阅读一些关于SSIM算法的文献来更好的理解算法，
    Image quality assessment: From error visibility to structural similarity
    然而及时你直接看下面的源代码，应该也能建立一个不错的映像。

    Scalar getMSSIM( const Mat& i1, const Mat& i2)
    {
     const double C1 = 6.5025, C2 = 58.5225;
     /***************************** INITS **********************************/
     int d     = CV_32F;// 4 字节 float

     Mat I1, I2;
     i1.convertTo(I1, d);           // 不能在单字节像素上进行计算，范围不够。
     i2.convertTo(I2, d);


     /***********************初步计算 ******************************/
     Mat I2_2   = I2.mul(I2);        // I2^2
     Mat I1_2   = I1.mul(I1);        // I1^2
     Mat I1_I2  = I1.mul(I2);        // I1 * I2

     Mat mu1, mu2;   //src img 高斯平滑（模糊） ksize - 核大小 Size(11, 11)  
     GaussianBlur(I1, mu1, Size(11, 11), 1.5);
     GaussianBlur(I2, mu2, Size(11, 11), 1.5);
     Mat mu1_2   =   mu1.mul(mu1);
     Mat mu2_2   =   mu2.mul(mu2);
     Mat mu1_mu2 =   mu1.mul(mu2);

    // src*src img 高斯平滑（模糊） ksize - 核大小 Size(11, 11)  
    // diff
     Mat sigma1_2, sigma2_2, sigma12;
     GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
     sigma1_2 -= mu1_2;

     GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
     sigma2_2 -= mu2_2;

     GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
     sigma12 -= mu1_mu2;

     //=======公式=======================
     Mat t1, t2, t3;

     t1 = 2 * mu1_mu2 + C1;
     t2 = 2 * sigma12 + C2;
     t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

     t1 = mu1_2 + mu2_2 + C1;
     t2 = sigma1_2 + sigma2_2 + C2;
     t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

     Mat ssim_map;
     divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

     Scalar mssim = mean( ssim_map ); // mssim = ssim_map的平均值
     return mssim;
    }

    这个操作会针对图像的每个通道返回一个相似度，取值范围应该在0到1之间，取值为1时代表完全符合。
    然而尽管SSIM能产生更优秀的数据，但是由于高斯模糊很花时间，
    所以在一个实时系统（每秒24帧）中，人们还是更多地采用PSNR算法。

    正是这个原因，最开始的源码里，我们用PSNR算法去计算每一帧图像，
    而仅当PSNR算法计算出的结果低于输入值的时候，用SSIM算法去验证。
    
=======================================================
# 用OpenCV创建视频
    使用OpenCV中的 VideoWriter 类就可以简单的完成创建视频的工作。
        如何用OpenCV创建一个视频文件
        用OpenCV能创建什么样的视频文件
        如何释放视频文件当中的某个颜色通道


    视频文件的结构

    首先，你需要知道一个视频文件是什么样子的。每一个视频文件本质上都是一个容器，
    文件的扩展名只是表示容器格式（例如 avi ， mov ，或者 mkv ）而不是视频和音频的压缩格式。
    容器里可能会有很多元素，例如视频流，音频流和一些字幕流等等。
    这些流的储存方式是由每一个流对应的编解码器(codec)决定的。
    通常来说，音频流很可能使用 mp3 或 aac 格式来储存。
    而视频格式就更多些，通常是 XVID ， DIVX ， H264 或 LAGS (Lagarith Lossless Codec)等等。
    具体你能够使用的编码器种类可以在操作系统的编解码器列表里找到。

    OpenCV能够处理的视频只剩下 avi 扩展名的了。
    另外一个限制就是你不能创建超过2GB的单个视频，
    还有就是每个文件里只能支持一个视频流，
    不能将音频流和字幕流等其他数据放在里面。

    找一些专门处理视频的库例如 FFMpeg 或者更多的编解码器例如 
    HuffYUV ， CorePNG 和 LCL 。
    你可以先用OpenCV创建一个原始的视频流然后通过其他编解码器转换成其他格式
    并用VirtualDub 和 AviSynth 这样的软件去创建各种格式的视频文件.

    要创建一个视频文件，你需要创建一个 VideoWriter 类的对象。
    可以通过构造函数里的参数或者在其他合适时机使用 open 函数来打开，两者的参数都是一样的：

    我们会使用输入文件名+通道名( argv[2][0])+avi来创建输出文件名。

    const string source      = argv[1];            // 原视频文件名
    string::size_type pAt = source.find_last_of('.');   // 找到扩展名的位置
    const string NAME = source.substr(0, pAt) + argv[2][0] + ".avi";   // 创建新的视频文件名

    // 编解码器
    VideoCapture inputVideo(source);                                   // 打开视频输入
    int ex = static_cast<int>(inputVideo.get(CV_CAP_PROP_FOURCC));     // 得到编码器的int表达式

    OpenCV内部使用这个int数来当作第二个参数，
    这里会使用两种方法来将这个 整型数转换为字符串：位操作符和联合体。
    前者可以用&操作符并进行移位操作，以便从int里面释放出字符：
    // 位操作符
    char EXT[] = {ex & 0XFF , (ex & 0XFF00) >> 8,(ex & 0XFF0000) >> 16,(ex & 0XFF000000) >> 24, 0};

    // 联合体 使用 联合体 来做到:
        union { int v; char c[5];} uEx ;
        uEx.v = ex;  // 通过联合体来分解字符串
        uEx.c[4]='\0';
        反过来，当你需要修改视频的格式时，你都需要修改FourCC码，
        而更改ForCC的时候都要做逆转换来指定新类型。如果你已经知道这个FourCC编码的具体字符的话，
        可以直接使用 *CV_FOURCC* 宏来构建这个int数:

    // 输出视频的帧率，也就是每秒需要绘制的图像数 inputVideo.get(CV_CAP_PROP_FPS)
    VideoWriter outputVideo;
    Size S = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),    //获取输入尺寸
                  (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

    outputVideo.open(NAME , ex, inputVideo.get(CV_CAP_PROP_FPS),S, true);
    // 然后，最好使用 isOpened() 函数来检查是不是成功打开。

    outputVideo.write(res);  //或者
    outputVideo << res;


    要“释放”出某个通道，又要保持视频为彩色，实际上也就意味着要把未选择的通道都设置为全0。
    这个操作既可以通过手工扫描整幅图像来完成，又可以通过分离通道然后再合并来做到，
    具体操作时先分离三通道图像为三个不同的单通道图像，
    然后再将选定的通道与另外两张大小和类型都相同的黑色图像合并起来。

    split(src, spl);                            // 分离三个通道
    for( int i =0; i < 3; ++i)
       if (i != channel)
          spl[i] = Mat::zeros(S, spl[0].type());//创建相同大小的黑色图像
    merge(spl, res);                            //重新合并

====================================
# 用GDAL读地理栅格文件
    GDAL(Geospatial Data Abstraction Library)是一个在X/MIT许可协议下的开源栅格空间数据转换库
    它利用抽象数据模型来表达所支持的各种文件格式。它还有一系列命令行工具来进行数据转换和处理。
    OGR是GDAL项目的一个分支，功能与GDAL类似，只不过它提供对矢量数据的支持。

    GDAL/OGR 快速入门
    http://live.osgeo.org/zh/quickstart/gdal_quickstart.html

     说下我的构思吧，opencv库里有很多关于数字图像处理的函数，
    但是它却局限于遥感图像的读取，而GDAL却对遥感影像的读取支持的很好，
    所有我想用GDAL将遥感影像读入，转成矩阵，传递到opencv中，然后使用opencv的函数来处理，
    不知道这个想法怎么样，还希望各位能指点。

        cv::Mat image = cv::imread(argv[1], cv::IMREAD_LOAD_GDAL | cv::IMREAD_COLOR );
        // load the dem model
        cv::Mat dem = cv::imread(argv[2], cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH );
        // create our output products
        cv::Mat output_dem(   image.size(), CV_8UC3 );
        cv::Mat output_dem_flood(   image.size(), CV_8UC3 );

====================================
# Using Kinect and other OpenNI compatible depth sensors 

    VideoCapture capture(0); //  VideoCapture capture( CAP_OPENNI );

    capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_VGA_30HZ );
    cout << "FPS    " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FPS ) << endl;

    for(;;)
    {
        Mat depthMap;
        Mat bgrImage;
        capture.grab();
        capture.retrieve( depthMap, CAP_OPENNI_DEPTH_MAP );
        capture.retrieve( bgrImage, CAP_OPENNI_BGR_IMAGE );
        if( waitKey( 30 ) >= 0 )
            break;
    }

==========================================
# Intel
# Using Creative Senz3D and other Intel Perceptual Computing SDK compatible depth sensors 


    VideoCapture capture(CAP_INTELPERC);

    VideoCapture capture( CAP_INTELPERC );
    capture.set( CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_PROFILE_IDX, 0 );
    cout << "FPS    " << capture.get( CAP_INTELPERC_DEPTH_GENERATOR+CAP_PROP_FPS ) << endl;


    for(;;)
    {
        Mat depthMap;
        Mat image;
        Mat irImage;
        capture.grab();
        capture.retrieve( depthMap, CAP_INTELPERC_DEPTH_MAP );
        capture.retrieve(    image, CAP_INTELPERC_IMAGE );
        capture.retrieve(  irImage, CAP_INTELPERC_IR_MAP);
        if( waitKey( 30 ) >= 0 )
            break;
    }



