#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;

int main(void)
{
	CascadeClassifier trafficLightCascader;
	string Cascade_name = "TrafficLight.xml";

	if (!trafficLightCascader.load(Cascade_name))
	{
		cout<<"Can't load the face feature data"<<endl;
		return -1;
	}
	
	vector<Rect> trafficLights;

	//离线图片
	ifstream imfile("E://TL//pics.txt");
	//char read_flag[100];
	string read_flag;

	while(getline(imfile, read_flag))
	{

		//离线图片
		//imfile>>read_flag;

		Mat src = imread(read_flag, -1);	//-1，原始图片，不改变其深度和通道数
		CvRect AssignRect = Rect(0, 0, src.cols, src.rows/2);
		Mat srcImage = src(AssignRect);
				
		Mat grayImage(srcImage.rows, srcImage.cols, CV_8UC1);

		cvtColor(srcImage, grayImage, CV_BGR2GRAY);
		equalizeHist(grayImage, grayImage);	//直方图均值化

		trafficLightCascader.detectMultiScale(grayImage, trafficLights, 1.1, 1, CV_HAAR_SCALE_IMAGE | CV_HAAR_FEATURE_MAX, Size(3, 3));
		//trafficLightCascader.detectMultiScale(grayImage, trafficLights, 1.1, 3, 0, Size(3,3));
		//detectMultiScale()的cvSize参数表示：寻找交通灯的最小区域。设置这个参数过大，会以丢失小物体为代价减少计算量

		for (int i=0; i<trafficLights.size(); ++i)
		{
			rectangle(src, trafficLights[i], Scalar(0, 255, 0), 2, 8, 0);
		}

		imshow("src", src);
		waitKey(100);
	}

	return 0;
}
