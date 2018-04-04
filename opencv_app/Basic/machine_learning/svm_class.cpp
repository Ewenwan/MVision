/*
使用 级联分类器 检测出人脸
缩放到 统一大小  放入
人工神经网络内 训练
笑脸 / 非笑脸
保存分类器

OpenCV 中使用的激活函数是另一种形式，
f(x) = b *  (1 - exp(-c*x)) / (1 + exp(-c*x))
当 α = β = 1 时
f(x) =(1 - exp(-x)) / (1 + exp(-x))

*/
#include "svm_class.h"

CascadeClassifier* faceCascade;  //人脸检测 级联分类器 指针
// Ptr<ANN_MLP> neuralNetwork;      //人工神经网络
// Ptr<SVM> svm = SVM::create();// SVM分类器
Ptr<SVM> svm_ptr;

//初始化一个用于训练的SVM
//参数：无
//返回：0 - 正常，-1 - 异常
int SVMInit(void)
{
	/*加载人脸检测分类器*/
	faceCascade = new CascadeClassifier();
	if(!faceCascade->load("../../common/data/cascade/haarcascades/haarcascade_frontalface_alt.xml"))
	{
		printf("Can't load face cascade.\n");
		return -1;
	}

	// SVM分类器参数
    svm_ptr = SVM::create();// 创建
    // 50*50 pix像素的图片 展开成一列 2500 像素输入 
    svm_ptr->setType(SVM::C_SVC);   //  SVM类型.  C_SVC 该类型可以用于n-类分类问题 (n \geq 2)
    svm_ptr->setC(0.5);             //  错分类样本离同类区域的距离 的权重
    svm_ptr->setKernel(SVM::LINEAR);   //  
    // 算法终止条件.   最大迭代次数和容许误差
    svm_ptr->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6)); 
	return 0;
}

//初始化一个用于分类的 SVM
//参数：训练好的 SVM分类器
//返回：0 - 正常，-1 - 异常
int SVMInit(const char *fileName)
{
	// 加载人脸检测 分类器
    faceCascade = new CascadeClassifier();
// ../../common/data/cascade/lbpcascades/lbpcascade_frontalface.xml  // lbp 特征
    if(!faceCascade->load("../../common/data/cascade/haarcascades/haarcascade_frontalface_alt.xml"))
    {
        printf("Can't load face cascade.\n");
        return -1;
    }
	/*从指定文件中加载神经网络数据*/
    printf("Loading SVM form %s\n",fileName);
    svm_ptr = Algorithm::load<SVM>(fileName);
// Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::load("ann_param");
    return 0;
}

//训练 SVM 并保存到指定的文件
//参数：正样本容器，负样本容器，保存 SVM 数据的文件名
//返回：0 - 正常，-1 - 异常
int SVMTrain(std::vector<Mat>* posFaceVector,std::vector<Mat>* negFaceVector, const char *fileName)
{
//初始化训练数据矩阵和类标矩阵====================
	//训练数据矩阵每行为一个样本，行数=正样本数+负样本数，列数=样本维数=样本宽度x样本高度
	//类标矩阵每行为一个样本的类标，行数同上，列数=1 +1代表正样本 -1代表负样本
    Mat trainDataMat(posFaceVector->size() + negFaceVector->size(), 2500, CV_32FC1);// 数据量 × 2500维度
// SVM 标签不能为 float or double
    Mat labelMat = Mat::zeros(posFaceVector->size() + negFaceVector->size(),1,CV_32S);//标签 数据量 × 1维度

// 生成训练数据矩阵和类标矩阵================
    for(unsigned int i = 0;i < posFaceVector->size();i++)
    {
      // 进来的图像为矩阵 reshap 到 一列 在复制到 trainDataMat的一行 并且 归一化到 0~1 之间  防止计算溢出
        posFaceVector->at(i).reshape(0,1).convertTo(trainDataMat.row(i),CV_32FC1,0.003921569,0);  //0.003921569=1/255
        labelMat.at<float>(i) = 1;//类标签 样本  1代表正样本  
    }

    for(unsigned int i = 0;i < negFaceVector->size();i++)
    {
        negFaceVector->at(i).reshape(0,1).convertTo(trainDataMat.row(i + posFaceVector->size()),CV_32FC1,0.003921569,0);
        labelMat.at<float>(i + posFaceVector->size()) = -1;// 代表负样本
    }

// 训练网络=================
    svm_ptr->train(trainDataMat,ROW_SAMPLE,labelMat);

// 保存网络数据=================
    svm_ptr->save(fileName);

    return 0;
}

//用训练好的神经网络进行分类
//参数：待分类的样本
//返回：1 - 正样本（笑脸），0 - 负样本（非笑脸），-1 - 未检测到人脸
int SVMClassify(Mat *image, Rect& face_dec)
{
	// 人脸检测
    Mat gray,roi,classifyDataMat;
    std::vector<Rect> faces;// 级联分类器 检测到的人脸矩阵 检测到的人脸 矩形区域 左下点坐标 长和宽
    cvtColor(*image,gray,CV_BGR2GRAY);   //将样本转换为灰度图
    //equalizeHist(gray,gray);             //直方图均衡化

    faceCascade->detectMultiScale(
            gray,
            faces,
            1.1,// 每张图像缩小的尽寸比例 1.1 即每次搜索窗口扩大10%
            4,// 每个候选矩阵应包含的像素领域
            CASCADE_SCALE_IMAGE|CASCADE_FIND_BIGGEST_OBJECT|CASCADE_DO_ROUGH_SEARCH,
            Size(50,50)); //在样本中检测人脸，只检测最大的人脸  至少50*50 像素的 表示最小的目标检测尺寸

    if(faces.size() < 1)                 //样本中检测不到人脸
        return -1;
    face_dec = faces[0];// 人脸
    roi = gray(faces[0]);// 取灰度图像的 第一个人脸
// Point center( faces[0].x + faces[0].width*0.5, faces[0].y + faces[0].height*0.5 );//中心点
    resize(roi,roi,Size(50,50),0,0,CV_INTER_LINEAR);  //将检测到的人脸区域缩放到50x50像素

// 对检测到的人脸进行分类，即判断是笑脸还是非笑脸 
    roi = Mat_<float>(roi);
    roi.reshape(0,1).convertTo(classifyDataMat,CV_32FC1,0.003921569,0);//转换为待分类数据  50×50---> 1× 2500 归一化到 0~1 
    float response = svm_ptr->predict(classifyDataMat);
    cout<< response << endl;  //打印分类结果 -1   1
    if(response >= 1)
        return 1;  //正类 1
    else
        return 0;  //否则为负样本（非笑脸）
}

//从指定图像中检测人脸
//参数：图像文件名，存放人脸区域的矩阵
//返回：1 - 成功检测到人脸，0 - 未检测到人脸，-1 - 异常
int detectFaceFormImg(const char *imgFilename, Mat *faceRoi)
{
    Mat img,gray,roi;
	
    std::vector<Rect> faces;// 级联分类器 检测到的人脸矩阵 检测到的人脸 矩形区域 左下点坐标 长和宽
	/*读取图像文件*/
	img = imread(imgFilename);
	if(img.empty())
	{
		printf("Can't read file %s\n",imgFilename);
		return -1;
	}

	cvtColor(img,gray,CV_BGR2GRAY);  //将图像转换为灰度图
	//equalizeHist(gray,gray);//直方图均衡化

	faceCascade->detectMultiScale(
			gray,
                        faces,
			1.1,// 每张图像缩小的尽寸比例 1.1 即每次搜索窗口扩大10%
			6,// 每个候选矩阵应包含的像素领域
			CASCADE_SCALE_IMAGE|CASCADE_FIND_BIGGEST_OBJECT|CASCADE_DO_ROUGH_SEARCH,
			Size(40,40));//在图像中检测人脸 表示最小的目标检测尺寸

    if(faces.size() < 1)             //未检测到人脸
	{
		printf("Can't find face in %s, skip it.\n",imgFilename); 
		return 0;
	}

    roi = gray(faces[0]);                           //人脸区域ROI
    resize(roi,roi,Size(50,50),0,0,CV_INTER_LINEAR);//将人脸区域ROI缩放到50x50像素

    *faceRoi = roi.clone();                         //将人脸区域ROI拷贝到输出矩阵

    return 1;
}
