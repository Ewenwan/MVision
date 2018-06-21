// Tencent is pleased to support the open source community by making ncnn available.
// https://opensource.org/licenses/BSD-3-Clause
// auther 万有文
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>// putText()
#include "net.h"

#include <sys/time.h>
#include <unistd.h>

// 计时
long getTimeUsec()
{
    
    struct timeval t;
    gettimeofday(&t,0);
    return (long)((long)t.tv_sec*1000*1000 + t.tv_usec);
}

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;// 前向模型
    squeezenet.load_param("squeezenet_v1.1.param");// 模型框架
    squeezenet.load_model("squeezenet_v1.1.bin");// 权重参数
    // 图片变形
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);
    // 各个通道均值
    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);// 图像减去均值归一化

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.set_light_mode(true);// 模型 提取器 

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);//提取 prob层的输出

    cls_scores.resize(out.w);
    for (int j=0; j<out.w; j++)
    {
        cls_scores[j] = out[j];//结果的每一个值 变成 vector 
    }

    return 0;
}
// 打印结果  有修改 保存类别结果和得分
static int print_topk(const std::vector<float>& cls_scores, int topk, std::vector<int>& index_result, std::vector<float> score_result)
{
    // partial sort topk with index
    int size = cls_scores.size();// 结果维度
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (unsigned int i=0; i<size; i++)
    {// 成对 值:id 这里id对于类别
        vec[i] = std::make_pair(cls_scores[i], i);
    }
    // 排序
    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;//得分值
        int index = vec[i].second;// id
        //fprintf(stderr, "%d = %f\n", index, score);
       //  添加保存结果的 vector数组
        score_result.push_back(score);
        index_result.push_back(index);
    }

    return 0;
}

// 添加一个载入 类别目录的文件 的 函数
static int load_labels(std::string path, std::vector<std::string>& labels)
{
    FILE* fp = fopen(path.c_str(), "r");//读取文件
    
    while(!feof(fp))
    {
      char str_b[1024];//先读取 1024个字符
      fgets(str_b, 1024, fp);
      std::string str_block(str_b);//转换成 string 方便操作
      
      if(str_block.length() > 0)
      {
        for (unsigned int i = 0; i <  str_block.length(); i++)
        {
           if(str_block[i] == ' ')
           {
              std:: string name = str_block.substr(i, str_block.length() - i - 1);
              labels.push_back(name);
              i = str_block.length();
           }
        }
      }
    }
   return 0 ;
}


int main(int argc, char** argv)
{
   // 命令行传入的图片文件名
    const char* imagepath = argv[1];
   // opencv 读取图像
    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
     
    // 读取类别标签文件
    std::vector<std::string> labels;
    load_labels("synset_words.txt", labels);
    
    std::vector<float> cls_scores;

    long time = getTimeUsec();
    detect_squeezenet(m, cls_scores);
    time = getTimeUsec() - time;
    printf("detection time: %ld ms\n",time/1000);


    std::vector<int> index;
    std::vector<float> score;

    print_topk(cls_scores, 3, index, score);

   for(unsigned int i = 0; i < index.size(); i++)
   {
     cv::putText(m, labels[index[i]], cv::Point(50, 50+30*i), CV_FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 100, 200), 2, 8);
   }
   
   cv::imshow("result", m);
   cv::imwrite("test_result.jpg", m);
   cv::waitKey(0);

   return 0;


}

