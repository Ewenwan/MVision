#include <iostream>
//双目图像处理
#include "stereoprocessor.h"
// 参数文件解析
#include <libconfig.h++>
// 图像预处理
#include "imageprocessor.h"
#include <cstdlib>
// boost 
#include <boost/filesystem.hpp>
#include <boost/date_time.hpp>
#include <iomanip>

// pcl库
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>

using namespace std;
using namespace cv;

// 读取文件夹下　照片文件名到vector中
bool loadImageList(string file, vector<string> &list)
{
    bool loadSuccess = false;
    FileNode fNode;
    FileStorage fStorage(file, FileStorage::READ);//读取

    if (fStorage.isOpened())
    {
        fNode = fStorage.getFirstTopLevelNode();

        if (fNode.type() == FileNode::SEQ)
        {
            for(FileNodeIterator iterator = fNode.begin(); iterator != fNode.end(); ++iterator)
            {
                list.push_back((string) *iterator);//每一个文件名
            }
            loadSuccess = true;
        }
    }

    return loadSuccess;
}
// 读取照片文件名vector　到图像Mat Vector中
bool loadImages(vector<string> fileList, vector<Mat> &images)
{
    bool emptyImage = false;
    for (int i = 0; i < fileList.size() && !emptyImage; ++i)
    {
        Mat curImg = imread(fileList[i]);//读取图片
        if (!curImg.empty())
        {
            images.push_back(curImg);
        }
        else
        {
            emptyImage = true;
        }
    }
    return (!emptyImage && images.size() > 0);
}
//  射影矩阵 
bool loadQMatrix(string file, Mat &Q)
{
    bool success = false;
    try
    {
        FileStorage fStorage(file.c_str(), FileStorage::READ);
        fStorage["Q"] >> Q;
        fStorage.release();
        success = true;
    }
    catch(Exception ex)
    {
    }
    return success;
}
//创建并保存点云
void createAndSavePointCloud(Mat &disparity, Mat &leftImage, Mat &Q, string filename)
{
    pcl::PointCloud<pcl::PointXYZRGB> pointCloud;

    // Read out Q Values for faster access
    double Q03 = Q.at<double>(0, 3);
    double Q13 = Q.at<double>(1, 3);
    double Q23 = Q.at<double>(2, 3);
    double Q32 = Q.at<double>(3, 2);
    double Q33 = Q.at<double>(3, 3);

    for (int i = 0; i < disparity.rows; i++)
    {
        for (int j = 0; j < disparity.cols; j++)
        {
            // Create a new point
            pcl::PointXYZRGB point;

            // Read disparity
            float d = disparity.at<float>(i, j);
            if ( d <= 0 ) continue; //Discard bad pixels

            // Read color
            Vec3b colorValue = leftImage.at<Vec3b>(i, j);
            point.r = static_cast<int>(colorValue[2]);
            point.g = static_cast<int>(colorValue[1]);
            point.b = static_cast<int>(colorValue[0]);

            // Transform 2D -> 3D and normalise to point
            double x = Q03 + j;
            double y = Q13 + i;
            double z = Q23;
            double w = (Q32 * d) + Q33;
            point.x = -x / w;
            point.y = -y / w;
            point.z = z / w;

            // Put point into the cloud
            pointCloud.points.push_back (point);
        }
    }
    // Resize PCL and save to file
    pointCloud.width = pointCloud.points.size();
    pointCloud.height = 1;
    pcl::io::savePCDFileASCII(filename, pointCloud);
}

int main(int argc, char *argv[])
{
    bool readSuccessfully = false;
    libconfig::Config cfg;

    bool success = false;

    string xmlImages, ymlExtrinsic;
//  ADCensus 算法参数
    uint dMin; uint dMax; Size censusWin;// 最小最大视差　Minimum and maximum disparity　窗口大小
    float defaultBorderCost;// ADCensus = AD＋Ｃensus
    float lambdaAD; float lambdaCensus; string savePath; uint aggregatingIterations;
    uint colorThreshold1; uint colorThreshold2; uint maxLength1; uint maxLength2; uint colorDifference;
    float pi1; float pi2; uint dispTolerance; uint votingThreshold; float votingRatioThreshold;
    uint maxSearchDepth; uint blurKernelSize; uint cannyThreshold1; uint cannyThreshold2; uint cannyKernelSize;
//cfg参数配置文件参数读取
    if(argc == 2)
    {
        try
        {
            cfg.readFile(argv[1]);//读取配置文件
            readSuccessfully = true;
        }
        catch(const libconfig::FileIOException &fioex)//文件读写错误
        {
            cerr << "[ADCensusCV] I/O error while reading file." << endl;
        }
        catch(const libconfig::ParseException &pex)//文件解析错误
        {
            cerr << "[ADCensusCV] Parsing error" << endl;
        }

        if(readSuccessfully)
        {
            try
            {
                dMin = (uint) cfg.lookup("dMin");//最小视差
                dMax = (uint) cfg.lookup("dMax");//最大视差
                xmlImages = (const char *) cfg.lookup("xmlImages");
                ymlExtrinsic = (const char *) cfg.lookup("ymlExtrinsic");//内参数文件
                censusWin.height = (uint) cfg.lookup("censusWinH");//census窗口  在指定窗口内比较周围亮度值与中心点的大小
                censusWin.width = (uint) cfg.lookup("censusWinW");
                defaultBorderCost = (float) cfg.lookup("defaultBorderCost");//权重
                lambdaAD = (float) cfg.lookup("lambdaAD"); // TODO Namen anpassen 信息结合权重
                lambdaCensus = (float) cfg.lookup("lambdaCensus");// 　cost = r(Cad , lamdAD) + r(Cces, lamdCensus)
                savePath = (const char *) cfg.lookup("savePath");
                aggregatingIterations = (uint) cfg.lookup("aggregatingIterations");// 自适应窗口代价聚合 次数
                colorThreshold1 = (uint) cfg.lookup("colorThreshold1");//搜索　自适应框时的颜色阈值
                colorThreshold2 = (uint) cfg.lookup("colorThreshold2");
                maxLength1 = (uint) cfg.lookup("maxLength1");//搜索范围阈值
                maxLength2 = (uint) cfg.lookup("maxLength2");
                colorDifference = (uint) cfg.lookup("colorDifference");
                pi1 = (float) cfg.lookup("pi1");
                pi2 = (float) cfg.lookup("pi2");
                dispTolerance = (uint) cfg.lookup("dispTolerance");
                votingThreshold = (uint) cfg.lookup("votingThreshold");
                votingRatioThreshold = (float) cfg.lookup("votingRatioThreshold");
                maxSearchDepth = (uint) cfg.lookup("maxSearchDepth");
                blurKernelSize = (uint) cfg.lookup("blurKernelSize");
                cannyThreshold1 = (uint) cfg.lookup("cannyThreshold1");
                cannyThreshold2 = (uint) cfg.lookup("cannyThreshold2");
                cannyKernelSize = (uint) cfg.lookup("cannyKernelSize");
            }
            catch(const libconfig::SettingException &ex)//未读取到部分参数
            {
                cerr << "[ADCensusCV] " << ex.what() << endl
                     << "config file format:\n"
                        "dMin(uint)\n"
                        "xmlImages(string)\n"
                        "ymlExtrinsic(string)\n"
                        "censusWinH(uint)\n"
                        "censusWinW(uint)\n"
                        "defaultBorderCost(float)\n"
                        "lambdaAD(float)\n"
                        "lambdaCensus(float)\n"
                        "savePath(string)\n"
                        "aggregatingIterations(uint)\n"
                        "colorThreshold1(uint)\n"
                        "colorThreshold2(uint)\n"
                        "maxLength1(uint)\n"
                        "maxLength2(uint)\n"
                        "colorDifference(uint)\n"
                        "pi1(float)\n"
                        "pi2(float)\n"
                        "dispTolerance(uint)\n"
                        "votingThreshold(uint)\n"
                        "votingRatioThreshold(float)\n"
                        "maxSearchDepth(uint)\n"
                        "blurKernelSize(uint)\n"
                        "cannyThreshold1(uint)\n"
                        "cannyThreshold2(uint)\n"
                        "cannyKernelSize(uint)\n";
                readSuccessfully = false;
            }
        }

        if(readSuccessfully)
        {
            vector<string> fileList;//图像文件列表容器
            vector<Mat> images;//　图像Mat容器
            Mat Q(4, 4, CV_64F);//　射影矩阵 

            boost::filesystem::path dir(savePath);//保存路径
            boost::filesystem::create_directories(dir);

            bool gotExtrinsic = loadQMatrix(ymlExtrinsic, Q);
            if(loadImageList(xmlImages, fileList))
            {
                if(loadImages(fileList, images))//图像Mat
                {
                    if (images.size() % 2 == 0)
                    {
                        bool error = false;
                        for (int i = 0; i < (images.size() / 2) && !error; ++i)
                        {
                            stringstream file;
                            file << savePath << i;
                            boost::posix_time::ptime start = boost::posix_time::second_clock::local_time();
                            boost::posix_time::ptime end;
                            boost::posix_time::time_duration diff;

                            ImageProcessor iP(0.1);// 图像预处理类  像素数量阈值比率
                            Mat eLeft, eRight;
// 3*3 ksize - 核大小  1.9 * images - gauss(images )　　高斯平滑　叠加
                            eLeft = iP.unsharpMasking(images[i * 2], "gauss", 3, 1.9, -1);
                            eRight = iP.unsharpMasking(images[i * 2 + 1], "gauss", 3, 1.9, -1);
// 双目处理类　ADcensus代价初始化　+　自适应窗口代价聚合　+　扫描线全局优化　+ 代价转视差, 外点(遮挡点+不稳定)检测  + 
// 视差传播－迭代区域投票法 使用临近好的点为外点赋值　      + 
// 视差传播-16方向极线插值（对于区域内点数量少的　外点　再优化） +
// candy边缘矫正 + 亚像素求精,中值滤波 
		StereoProcessor sP(	dMin, dMax, images[i * 2], images[i * 2 + 1], 
					censusWin, defaultBorderCost, lambdaAD, 	
					lambdaCensus, file.str(), aggregatingIterations, 
					colorThreshold1, colorThreshold2, maxLength1, maxLength2,
				        colorDifference, pi1, pi2, dispTolerance, votingThreshold, 
					votingRatioThreshold,maxSearchDepth, blurKernelSize, 
					cannyThreshold1, cannyThreshold2, cannyKernelSize);
                            string errorMsg;
                            error = !sP.init(errorMsg);//参数检测

                            if(!error && sP.compute())//计算视差
                            {
                                success = true;
                                if(gotExtrinsic)
                                {
                                    Mat disp = sP.getDisparity();//得到视差

                                    string dispFile = file.str();//视差文件名
                                    dispFile += "_disp.yml";
                                    FileStorage fs(dispFile, FileStorage::WRITE);//保存视差文件
                                    fs << "disp" << disp;
                                    fs.release();

                                    file << "_cloud.pcd";//点云文件
                                    createAndSavePointCloud(disp, images[i * 2], Q, file.str());
                                }
                                else
                                {
                                    cerr << "[ADCensusCV] Could not create point cloud (no extrinsic)!" << endl;
                                }
                            }
                            else
                            {
                                cerr << "[ADCensusCV] " << errorMsg << endl;
                            }

                            end = boost::posix_time::second_clock::local_time();

                            diff = end - start;

                            cout << "Finished computation after " << setw(2) << right <<  setfill('0') << ((int)(diff.total_seconds() / 3600)) << ":"
                                                                   << setw(2) << right <<  setfill('0') << ((int)((diff.total_seconds() / 60) % 60)) << ":"
                                                                   << setw(2) << right <<  setfill('0') << (diff.total_seconds() % 60) << " (" << diff.total_seconds() << "s) !" << endl;

                        }
                    }
                    else
                    {
                        cerr << "[ADCensusCV] Not an even image number!" << endl;
                    }
                }
                else
                {
                    cerr << "[ADCensusCV] Could not read images!" << endl;
                }
            }
            else
            {
                cerr << "[ADCensusCV] Could not read image list!" << endl;
            }
        }
    }
    else
    {
        cerr << "[ADCensusCV] ADCensusBM <config file>" << endl;
    }

    return (success)? EXIT_SUCCESS: EXIT_FAILURE;
}

