# 简易的图像标注小工具 图像分类/检测/语义 标注

[OpenCV探索之路（二十五）：制作简易的图像标注小工具](https://www.cnblogs.com/skyfsm/p/7613314.html)

## 图像分类标注小工具

实现图像分类的小工具太好开发了，因为它功能很简单，无非是对一个文件夹内的所有图片进行分类，生成每张图片所对应的类别标签，用txt文件存储起来，当然也可以把每一类图片放在对应的该类的文件夹下。

我实现的这个图像分类小工具的功能就是，循环弹出一个文件夹内所有的图片，标注人员对这张图片进行分类，属于1类就按1，属于2类就按2，如此类推，按完相应号码后图片自动跳到下一张，直至文件夹内的图片都被标注完毕。

我们以下面的图库为例，将其分为3类。
```c
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#define  DATA_DIR ".\\dataset\\"
#define  IMG_MAX_NUM  20


using namespace cv;
using namespace std;

int main()
{
    FILE* fp;
    FILE* fp_result;
    fp = fopen("start.txt", "r");  //读取开始的图片名字，方便从某一图片开始标注
    int start_i = 0;
    fscanf(fp, "%d", &start_i);
    fclose(fp);

    fp_result = fopen("classify_record.txt", "a+");   //用于记录每张图每个框的标注信息

    printf("start_i: %d\n", start_i);

    /*循环读取图片来标注*/
    for (int i = start_i; i < IMG_MAX_NUM; i++)
    {
        stringstream ss1,ss2,ss3;

        ss1 << DATA_DIR <<"data\\"<< i << ".jpg";
        ss3 << i << ".jpg";
        Mat src = imread(ss1.str());
        if (src.empty())
        {
            continue;
        }
        printf("正在操作的图像: %s\n", string(ss1.str()).c_str());
        
        imshow("标注", src);

        char c = 0;
        c = waitKey(0);
        while ( c != '1' && c != '2' && c != '3')  
        {
            c = waitKey(0);
            printf("invaid input!\n");
        }

        ss2 << DATA_DIR << c << "\\" << i << ".jpg";

        char type = c - '0';
        printf("分类为: %d\n", c - '0');  
        imwrite(ss2.str(), src);   //copy一份到对应类别的文件夹
        fprintf(fp_result, "%s %d\n", string(ss3.str()).c_str(), type);
    }
    

    fclose(fp_result);
    return 0;
}


```
## 目标检测图像标注小工具

我们做标注时不仅仅要把我们想要识别的物体用矩形框将其框出来，还需要记录这个框的相关信息，比如这个框的左顶点坐标、宽度高度等（x,y,w,h)。为了能实现这个标注任务，这个标注小工具必须具备框图和自动记录（x,y,w,h)信息的功能。

利用opencv我们可以快速实现用矩形框框出对应物体的功能，再加上将每个矩形框的信息有序记录在txt文件的功能，一个用于检测图像标注小工具就算开发好了。

```c
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>


#define  DATA_DIR ".\\cut256\\"
#define  IM_ROWS  5106
#define  IM_COLS  15106
#define  ROI_SIZE 256

using namespace cv;
using namespace std;

Point ptL, ptR; //鼠标画出矩形框的起点和终点,矩形的左下角和右下角
Mat imageSource, imageSourceCopy;
FILE* fp_result;


struct UserData
{
    Mat src;
    vector<Rect> rect;
};


void OnMouse(int event, int x, int y, int flag, void *dp)
{
    UserData *d = (UserData *)dp;
    imageSourceCopy = imageSource.clone();

    if (event == CV_EVENT_LBUTTONDOWN)  //按下鼠标右键，即拖动开始
    {
        ptL = Point(x, y);
        ptR = Point(x, y);
    }
    if (flag == CV_EVENT_FLAG_LBUTTON)   //拖拽鼠标右键，即拖动进行
    {
        ptR = Point(x, y);
        imageSourceCopy = imageSource.clone();
        rectangle(imageSourceCopy, ptL, ptR, Scalar(0, 255, 0));
        imshow("标注", imageSourceCopy);
        
    }
    if (event == CV_EVENT_LBUTTONUP)  //拖动结束
    {
        if (ptL != ptR)
        {
            rectangle(imageSourceCopy, ptL, ptR, Scalar(0, 255, 0));
            imshow("标注", imageSourceCopy);

            int h = ptR.y - ptL.y;
            int w = ptR.x - ptL.x;


            printf("选择的信息区域是:x:%d  y:%d  w:%d  h:%d\n", ptL.x, ptL.y, w, h);

            d->rect.push_back(Rect(ptL.x, ptL.y, w, h));
            //d->src(imageSourceCopy);
        }
    }

    //点击右键删除一个矩形
    if (event == CV_EVENT_RBUTTONDOWN)
    {
        if (d->rect.size() > 0)
        {
            Rect temp = d->rect.back();

            printf("删除的信息区域是:x:%d  y:%d  w:%d  h:%d\n", temp.x, temp.y, temp.width, temp.height);
            d->rect.pop_back();

            for (int i = 0; i < d->rect.size(); i++)
            {
                rectangle(imageSourceCopy, d->rect[i], Scalar(0, 255, 0), 1);
            }
                      
        }
    }

}


void DrawArea(Mat& src, string img_name, string path_name)
{
    Mat img = src.clone();
    char c = 'x';
    UserData d;
    d.src = img.clone();
    while (c != 'n')
    {
        Mat backup = src.clone();
        imageSource = img.clone();
        
        namedWindow("标注", 1);
        imshow("标注", imageSource);
        setMouseCallback("标注", OnMouse, &d);

        c = waitKey(0);

        if (c == 'a')
        {
            printf("rect size: %d\n", d.rect.size());
            for (int i = 0; i < d.rect.size(); i++)
            {
                rectangle(backup, d.rect[i], Scalar(0, 255, 0), 1);
            }

            img = backup.clone();
            
        }
    }

    fprintf(fp_result, "%s\n", img_name.c_str());
    fprintf(fp_result, "%d\n", d.rect.size());
    for (int i = 0; i < d.rect.size(); i++)
    {
        Rect t = d.rect[i];

        fprintf(fp_result, "%d %d %d %d\n", t.x, t.y, t.width, t.height);
    }

    imwrite(path_name, img);
    

}
int main()
{
    FILE* fp;
    fp = fopen("start.txt", "r");
    int start_i = 0;
    int start_j = 0;
    fscanf(fp, "%d %d", &start_i, &start_j);
    fclose(fp);

    fp_result = fopen("record.txt", "a+");

    printf("start_i: %d, start_j: %d\n", start_i, start_j);


    /*循环读取图片来标注*/
    for (int i = start_i; i< IM_ROWS / ROI_SIZE + 1; i++)
    {
        for (int j = start_j; j<IM_COLS / ROI_SIZE; j++)
        {
            stringstream ss1, ss2;

            ss1 << DATA_DIR << "2017\\" << i << "_" << j << "_" << ROI_SIZE << "_.jpg";
            ss2 << DATA_DIR << "label_img\\" << i << "_" << j << "_" << ROI_SIZE << "_.jpg";
            cout << ss1.str() << endl;
            string str(ss1.str());
            string str2(ss2.str());
            cv::Mat src = cv::imread(ss1.str());

            DrawArea(src, str,str2);

       
        }

    }
    fclose(fp_result);
    return 0;
}


```

## 语义分割图像标注小工具
语义分割的标注相比上面的标注要复杂得多，所以标注工具开发起来也略难一点。

比如有这么一个任务，我们需要把图像中的建筑物给标注出来，生成一个mask图。


我们以后就可以根据这些mask图作为label来进行语义分割网络的训练了。

实现这么一个工具还是不算太复杂，主要功能的实现就在于使用了opencv的多边形的生成与填充函数。标注人员只需要在要标注的物体边缘打点，然后工具就会自动填充该区域，进而生成黑白mask图。


```c
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

#define  DATA_DIR ".\\cut256\\"

#define  IM_ROWS  5106
#define  IM_COLS  15106
#define  ROI_SIZE 256
struct UserData
{
    cv::Mat src;
    vector<cv::Point> pts;
};

FILE* fpts_set;

void on_mouse(int event, int x, int y, int flags, void *dp)
{
    UserData *d = (UserData *)dp;
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        d->pts.push_back(cv::Point(x, y));
    }
    if (event == CV_EVENT_RBUTTONDOWN)
    {
        if (d->pts.size()>0)
            d->pts.pop_back();
    }
    cv::Mat temp = d->src.clone();
    if (d->pts.size()>2)
    {
        const cv::Point* ppt[1] = { &d->pts[0] };
        int npt[] = { static_cast<int>(d->pts.size()) };
        cv::fillPoly(temp, ppt, npt, 1, cv::Scalar(0, 0, 255), 16);

    }
    for (int i = 0; i<d->pts.size(); i++)
    {
        cv::circle(temp, d->pts[i], 1, cv::Scalar(0, 0, 255), 1, 16);
    }
    cv::circle(temp, cv::Point(x, y), 1, cv::Scalar(0, 255, 0), 1, 16);
    cv::imshow("2017", temp);

}

void WriteTxT(vector<cv::Point>& pst)
{
    for (int i = 0; i < pst.size(); i++)
    {
        fprintf(fpts_set, "%d %d", pst[i].x, pst[i].y);
        if (i == pst.size() - 1)
        {
            fprintf(fpts_set, "\n");
        }
        else
        {
            fprintf(fpts_set, " ");
        }
    }
}

int label_img(cv::Mat &src, cv::Mat &mask, string& name)
{
    char c = 'x';

    vector<vector<cv::Point> > poly_point_set;

    while (c != 'n')
    {
        UserData d;
        d.src = src.clone();

        cv::namedWindow("2017", 1);
        cv::setMouseCallback("2017", on_mouse, &d);
        cv::imshow("2017", src);
        c = cv::waitKey(0);
        if (c == 'a')
        {
            if (d.pts.size()>0)
            {
                const cv::Point* ppt[1] = { &d.pts[0] };
                int npt[] = { static_cast<int>(d.pts.size()) };
                cv::fillPoly(src, ppt, npt, 1, cv::Scalar(0, 0, 255), 16);
                cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255), 16);
                poly_point_set.push_back(d.pts);
            }


        }
    }

    fprintf(stdout, "%s %d\n", name.c_str(), poly_point_set.size());
    fprintf(fpts_set, "%s %d\n", name.c_str(), poly_point_set.size());

    //将点集写入文件
    for (int i = 0; i < poly_point_set.size(); i++)
    {
        WriteTxT(poly_point_set[i]);
    }

    return 0;
}
int main()
{
    FILE* fp;
    fp = fopen("start.txt", "r");
    int start_i = 0;
    int start_j = 0;
    fscanf(fp, "%d %d", &start_i, &start_j);
    fclose(fp);

    fpts_set = fopen("semantic_label.txt", "a+");

    printf("start_i: %d, start_j: %d\n", start_i, start_j);

    for (int i = start_i; i<IM_ROWS / ROI_SIZE + 1; i++)
    {
        for (int j = start_j; j<IM_COLS / ROI_SIZE; j++)
        {
            stringstream ss1,ss2,ss3;
            cv::Mat mask(256, 256, CV_8UC1);
            mask.setTo(0);

            ss1 << DATA_DIR << "2017\\" << i << "_" << j << "_" << ROI_SIZE << "_.jpg";
            ss2 << DATA_DIR << "label\\" << i << "_" << j << "_" << ROI_SIZE << "_.jpg";
            ss3 << i << "_" << j << "_" << ROI_SIZE << "_.jpg";
            cout << ss1.str() << endl;

            cv::Mat src = cv::imread(ss1.str());

            label_img(src, mask, string(ss3.str()));// label based on tiny
            cv::imwrite(ss2.str(), mask);
        }

    }

    fclose(fpts_set);
    return 0;
}



```



