/*

使用 级联分类器 检测出人脸
缩放到 统一大小  放入
SVM 训练
笑脸 / 非笑脸
保存分类器

OpenCV 中使用的激活函数是另一种形式，
f(x) = b *  (1 - exp(-c*x)) / (1 + exp(-c*x))
当 α = β = 1 时
f(x) =(1 - exp(-x)) / (1 + exp(-x))

使用 ./smile_dec -e   检测笑脸
./smile_dec -t        训练笑脸分类器

 

*/
#include <stdio.h>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>

#include "svm_class.h"

using namespace std;
using namespace cv;

void printHelpMessage(void);      //打印帮助信息
int  preview(void);               //预览摄像头图像
int  trainSVM(void);              //SVM
int  smileFaceDetect(void);       //在摄像头图像中执行笑脸检测

char devPath[256] = "/dev/video0";                       //摄像头设备文件名
char posPath[256] = "../trainset/positive/";             //正训练集路径
char negPath[256] = "../trainset/negtive/";              //负训练集路径
char SVMFilename[256] = "../networks/SVMdefault.xml";    //保存神经网络数据的文件名

bool preview_flag = false;  //预览标志
bool train_flag   = false;  //训练标志
bool detect_flag  = false;  //笑脸检测标志

std::vector<Mat> posFaces;  //存放正样本的 Mat 容器
std::vector<Mat> negFaces;  //存放负样本的 Mat 容器

//主函数
int main(int argc, char *argv[])
{
	/*命令行参数解析*/
	int opt;

	if(argc == 1) //没有参数
		printHelpMessage();

    while((opt = getopt(argc,argv,"hpted:T:F:N:")) != -1)//获取下一个命令行参数
	{
		switch(opt)
		{
			case 'h':printHelpMessage();    break; //打印帮助信息
			case 'd':strcpy(devPath,optarg);break; //指定摄像头设备
			case 'p':preview_flag = true;   break; //预览摄像头图像
			case 't':train_flag = true;     break; //训练神经网络
                        case 'e':detect_flag = true;    break; //从摄像头捕获视频中检测笑脸
			case 'T':strcpy(posPath,optarg);break; //指定正训练集路径
			case 'F':strcpy(negPath,optarg);break; //指定负训练集路径
                   case 'N':strcpy(SVMFilename,optarg); break; //指定神经网络文件名
			default:printf("Invalid argument %c!\n",opt);return -1;//非法参数
		}
	}

	/*如果路径字符串末位非'\'，则添加之*/
	if(posPath[strlen(posPath)-1] != '/')
		strcat(posPath,"/");
	if(negPath[strlen(negPath)-1] != '/')
		strcat(negPath,"/");

	/*根据标志决定程序的功能*/
	if(train_flag == true)
		trainSVM();       //训练SVM
	else if(preview_flag == true)
		preview();        //预览摄像头图像
        else if(detect_flag == true)
        	smileFaceDetect();//从摄像头捕获视频中检测笑脸

    usleep(300000);
    return 0;
}

//将摄像头设备名转换为设备索引
//参数：设备名字符串 /dev/video0 ~ /dev/video100
//返回：设备索引，范围0-9999
int getDeviceIndex(const char *devicePath)
{
	/*检查输入字符串是否为Linux下的V4L2设备名*/
	if(strncmp(devicePath,"/dev/video",10))
	{
		printf("Invalid device path!\n");
		return -1;
	}

	char devIndex[5];
	unsigned int i;

	/*截取"/dev/video"之后的子串*/
	for(i = 10;i < strlen(devicePath);i++)
		devIndex[i-10] = devicePath[i];
	devIndex[i-10] = '\0';
	/*将字符串转换为整形数并返回*/
	return atoi(devIndex);
}

//训练一个神经网络，并将网络数据保存到文件
//参数：无
//返回：0 - 正常，-1 - 异常
int trainSVM(void)
{
	SVMInit();               //初始化网络
	posFaces.reserve(1000);  //为容器预留1000个样本的内存
	negFaces.reserve(1000);

// 遍历正样本
	DIR *posDir = opendir(posPath);//打开 正训练集路径
	if(posDir == NULL)
	{
		printf("Can't open directory %s\n",posPath);
		perror(posPath);
		return -1;
	}
	struct dirent *posDirent;
	while((posDirent = readdir(posDir)) != NULL)//读取下一个文件入口
	{
		/*跳过两个特殊文件*/
		if(strcmp(posDirent->d_name,".") == 0 || strcmp(posDirent->d_name,"..") == 0)
			continue;
		/*获得当前文件的完整文件名*/
		char posFileName[256];
		strcpy(posFileName, posPath);
		strcat(posFileName, posDirent->d_name);

		//处理当前文件 
		Mat temp;
		int result;
		result = detectFaceFormImg(posFileName, &temp);  //对当前文件做人脸检测
		if(result == 1)
		posFaces.push_back(temp);                        //若检测成功，则将人脸添加到正样本容器中
	}
        printf("Got %d positive samples.\n",(int)posFaces.size());

	// 遍历负样本 
	DIR *negDir = opendir(negPath);//打开负训练集路径
	if(negDir == NULL)
	{
		printf("Can't open directory %s\n",negPath);
		perror(negPath);
		return -1;
	}
	struct dirent *negDirent;
	while((negDirent = readdir(negDir)) != NULL)     //读取下一个文件入口
	{
		/*跳过两个特殊文件*/
		if(strcmp(negDirent->d_name,".") == 0 || strcmp(negDirent->d_name,"..") == 0)
			continue;

		/*获得当前文件的完整文件名*/
		char negFileName[256];
		strcpy(negFileName,negPath);
		strcat(negFileName,negDirent->d_name);

		/*处理当前文件*/
		Mat temp;
		int result;
		result = detectFaceFormImg(negFileName,&temp);//对当前文件做人脸检测
		if(result == 1)
		negFaces.push_back(temp);                     //若检测成功，则将人脸添加到负样本容器
	}
        printf("Got %d negtive samples.\n",(int)negFaces.size());

	struct timeval before, after;//时间

	/*训练神经网络，并保存到文件*/
	printf("begin network training...\n");
	gettimeofday(&before,NULL);
	SVMTrain(&posFaces, &negFaces, SVMFilename);
	gettimeofday(&after,NULL);
	long int usec = (after.tv_sec - before.tv_sec)*1000000 + (after.tv_usec-before.tv_usec);
	printf("training completed,use %lds %ldms\n",usec/1000000,(usec%1000000)/1000);
	return 0;
}

//在摄像头捕获视频中检测笑脸
//参数：无
//返回：0 - 正常，-1 - 异常
int smileFaceDetect(void)
{
	SVMInit(SVMFilename);//初始化检测神经网络
	VideoCapture capture;
	capture.open(getDeviceIndex(devPath));//打开摄像头捕获
	if(!capture.isOpened())
	{
	printf("Can not open device!\n");
	return -1;
	}
        //设置捕获图像分辨率  320*240
	capture.set(CAP_PROP_FRAME_WIDTH, 640);     
	capture.set(CAP_PROP_FRAME_HEIGHT,480);

	Mat frame;
	while(waitKey(10) != 'q') //循环直到按下键盘上的Q键
	{
		capture>>frame;                 //捕获一帧图像
		Rect  face_dec;
		if(SVMClassify(&frame, face_dec) == 1)//如果当前帧中检测到笑脸，则绘制一个醒目的红色矩形框
		{
		    // rectangle(frame,Point(0,0),Point(319,239),Scalar(0,0,255),10,4,0);

	   	   Point center( face_dec.x + face_dec.width*0.5, face_dec.y + face_dec.height*0.5 );
	   	   ellipse( frame, center, Size(face_dec.width*0.5, face_dec.height*0.5), 0, 0, 360, Scalar( 0, 0, 255 ), 10, 4, 0 );
		}
		imshow("smile face detecting",frame);  //显示当前帧
	}

	capture.release();

	return 0;
}

//预览摄像头捕获图像
//参数：无
//返回：0 - 正常，-1 - 异常
int preview(void)
{
	VideoCapture capture;
	capture.open(getDeviceIndex(devPath));    //打开摄像头捕获
	if(!capture.isOpened())
	{
		printf("Can not open device!\n");
		return -1;
	}

	capture.set(CAP_PROP_FRAME_WIDTH,640);   //设置捕获图像分辨率
	capture.set(CAP_PROP_FRAME_HEIGHT,480);

	Mat frame;
	
	while(waitKey(10) != 'q')                //循环直到按下键盘上的Q键
	{
		capture>>frame;                      //捕获一帧图像
		imshow("real-time video",frame);     //显示当前帧
	}

	capture.release();

	return 0;
}

//打印帮助信息
//参数：无
//返回：无
void printHelpMessage(void)
{
	printf("A BP artificial neural network demo	for smile detection.\n");
	printf("Usage:\n");
	printf("\tsmileDetection [options] ...\n\n");
	printf("Options:\n");
	printf("\t-h\tprint this help message\n");
	printf("\t-p\tpreview the real-time video\n");
	printf("\t-d\tspecify the camera device, default /dev/video0\n");
	printf("\t-t\ttrain a network only\n");
    printf("\t-e\tdetect smile face form camera capture stream\n");
	printf("\t-T\tspecify the positive sample location, default ../trainset/positive/\n");
	printf("\t-F\tspecify the negtive sample location, default ../trainset/negtive/\n");
	printf("\t-N\tspecify the network file, default ../networks/default.xml\n");
}
