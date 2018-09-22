// Tencent is pleased to support the open source community by making ncnn available.
// 
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "net.h"

// 计时模块
#include <sys/time.h>
#include <unistd.h>

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;
    squeezenet.load_param("squeezenet_v1.1.param");
    squeezenet.load_model("squeezenet_v1.1.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);// 图像缩放到 227,227

    const float mean_vals[3] = {104.f, 117.f, 123.f};// 三通道均值
    in.substract_mean_normalize(mean_vals, 0);// 减去均值后,归一化

    ncnn::Extractor ex = squeezenet.create_extractor();// 模型

    ex.input("data", in);// 设置data 数据

    ncnn::Mat out;
    ex.extract("prob", out);// 获取 prob 输出

    cls_scores.resize(out.w);
    for (int j=0; j<out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }
   // partial_sort 部分排序
    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

// 打印结果  有修改 保存类别结果和得分
static int print_topk(const std::vector<float>& cls_scores, int topk, std::vector<int>& index_result, std::vector<float> score_result)
{
    // partial sort topk with index
    int size = cls_scores.size();// 结果维度
   //  std::cout << "detction " << size << " nums class " << std::endl;
    
 //   fprintf(stderr, "detction %d  nums class obj \n", size);
    
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for ( int i=0; i<size; i++)
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

// 打印结果  有修改 保存类别结果和得分
static int print_topk(const std::vector<float>& cls_scores, int topk, std::vector<std::string>& label)
{
    // partial sort topk with index
    int size = cls_scores.size();// 结果维度
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for ( int i=0; i<size; i++)
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
       fprintf(stderr, "%s = %f\n", label[index].c_str(), score); 
    }

    return 0;
}

// ============================================
// 添加 计时
long getTimeUsec()
{
    
    struct timeval t;
    gettimeofday(&t,0);
    return (long)((long)t.tv_sec*1000*1000 + t.tv_usec);
}

// 添加一个载入 类别目录的文件 的 函数==========
/* 文件示例
n01440764 tench, Tinca tinca
n01443537 goldfish, Carassius auratus
n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
n01491361 tiger shark, Galeocerdo cuvieri
n01494475 hammerhead, hammerhead shark
n01496331 electric ray, crampfish, numbfish, torpedo
n01498041 stingray
n01514668 cock
n01514859 hen
 */
static int load_labels(std::string path, std::vector<std::string>& labels)
{
    FILE* fp = fopen(path.c_str(), "r");// .c_str() 转换成c字符串  读取文件
    
    if(fp == NULL) 
    {
    fprintf(stderr, "%s file not exit \n", path.c_str());
    return -1;
    }
    
    while(!feof(fp))// 直到文件结尾  
    {
      char str_b[1024];//先读取 1024个字符
      char * un = fgets(str_b, 1024, fp);
      if(un==NULL) return -1;
      std::string str_block(str_b);//转换成 string 方便操作
      
      if(str_block.length() > 0)
      {
        for (unsigned int i = 0; i <  str_block.length(); i++)
        {
           if(str_block[i] == ' ')
           {
              std:: string name = str_block.substr(i, str_block.length() - i - 1);//第一个空格之后的子串
              labels.push_back(name);// 存储名字
              i = str_block.length();// 跳到下一个
           }
        }
      }
    }
    
  //  
   return 0 ;fprintf(stderr, "%d class \n",  (int)labels.size()); 
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }
  // 命令行传入的图片文件名
    const char* imagepath = argv[1];
   // opencv 读取图像
    cv::Mat pic= cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (pic.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

  // 读取类别标签文件
  std::vector<std::string> labels;
  load_labels("synset_words.txt", labels);
 // if(flag == -1) return -1;
  fprintf(stderr, "%d class \n",  (int)labels.size()); // 1000类 每一类都会有一个 概率
  
///*
    std::vector<float> cls_scores;
    
    long time = getTimeUsec();
    detect_squeezenet(pic, cls_scores);
    time = getTimeUsec() - time;
    printf("detection time: %ld ms\n",time/1000);

  // print_topk(cls_scores, 3);
//*/

 ///*
    // std::vector<int> index;
    // std::vector<float> score;
    // print_topk(cls_scores, 3, index, score);
      
    print_topk(cls_scores, 3, labels);
    
    return 0;
}
