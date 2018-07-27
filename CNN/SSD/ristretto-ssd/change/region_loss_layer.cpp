////////////////////////////////
// yolov2 损失层=============================

// 网络 输入 ================================================
// 数据 data ===========
// top[0] 为 数据区 data 
// top_data = batch->data_.mutable_cpu_data();
// n*3*416*416==========
// top[0]->num()       图片数量 N
// top[0]->channels()  通道数量 3
// top[0]->height()    尺寸 300/416
// top[0]->width();

// 标签 label ===========
// top[1] 为 标签区 label 需要以不同格子大小区分()=======13*13格子======
// top_label ： batch->multi_label_[i]->mutable_cpu_data() 
//   N*150 ==============
//  150 = 30*(1+4)  预设 30个边框空间，5个为一组边框参数，边框数量不足30， 后面填0============
//  1个类别id + 4个边框参数(x,y,w,h  (范围0~1))
// top[2]============26*26格子============

// 网络输出 ====================================================================================
// 图片输入  n*3*416*416 
// 多层隐含网络 下采样卷积提取特征
// 网络输出  n*(5*(5+class_num))*13*13 = n*125*13*13 / n*425*13*13

// loss层 ===============================================================================
// 层类型 RegionLoss
// 层名字 det_loss
// 输入1： 网络输出    n*125*13*13 / n*425*13*13  --> n*169*5*25 /  n*169*5*85
//                                                        4+1+20 /  4+1+80
// 输入2： 标签 label  N*150, 30个边框=====================

// 输出1： loss==============
// loss += real_diff[i] * real_diff[i];  差分梯度 平方和  就是差平方和损失函数
// top[0]->mutable_cpu_data()[0] = loss;
// loss 包含=====================================
// 1. 边框置信度 的  差分梯度
// 2. 类别概率   的  差分梯度
// 3. 预测边框 和 实际边框 的 差分梯度
// 4. 前 12800次迭代 的 预测边框 和 预设边框 的 差分梯度



#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
//#include "caffe/layers/detection_evaluate_layer.hpp"
#include "caffe/util/bbox_util.hpp"

int iter = 0;

namespace caffe {

template <typename Dtype>
Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
  Dtype left = std::max(x1 - w1 / 2, x2 - w2 / 2);
  Dtype right = std::min(x1 + w1 / 2, x2 + w2 / 2);
  return right - left;
}

template <typename Dtype>
Dtype Calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth) {
  Dtype w = Overlap(box[0], box[2], truth[0], truth[2]);
  Dtype h = Overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;
  Dtype inter_area = w * h;
  Dtype union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area / union_area;
}
///*
////template <typename Dtype>
//inline Dtype sigmoid(Dtype x)
//{
 // return 1. / (1. + exp(-x));
//}

template <typename Dtype>
void disp(Blob<Dtype>& swap)
{
  std::cout<<"#######################################"<<std::endl;
  for (int b = 0; b < swap.num(); ++b)
    for (int c = 0; c < swap.channels(); ++c)
      for (int h = 0; h < swap.height(); ++h)
      {
  	std::cout<<"[";
        for (int w = 0; w < swap.width(); ++w)
	{
	  std::cout<<swap.data_at(b,c,h,w)<<","; 
	}
	std::cout<<"]"<<std::endl;
      }
  return;
}
//*/
// 类别概率部分 归一化处理
// 类别概率  减去最大值 (-,0]) 在指数映射 -> (0,1] 后归一化
template <typename Dtype>
Dtype softmax_region(Dtype* input, int classes)
// Dtype* input  每个边框对每个类别的预测概率
// int classes   类别总数
{
  Dtype sum = 0;
  Dtype large = input[0];// 初始化最大值
  Dtype tep = input[10];// temp 临时
  //int id;
  
  // 遍历记录 类别预测概率 最大值
  for (int i = 0; i < classes; ++i)
  {
    if (input[i] > large)
      large = input[i];// 记录类别 概率中的最大值
      //id = i;// 最大值下标
  }
  
//std::cout<< "======large : " << large << std::endl;
  // 减最大值，后求和
  for (int i = 0; i < classes; ++i)
  {
    Dtype e = exp(input[i] - large);// 减去最大值 (-inf,0]) 在指数映射 -> (0,1]
    sum += e;// 求和
    input[i] = e;
  }
  
  for (int i = 0; i < classes; ++i)
  {
    input[i] = input[i] / sum;// 归一化   -inf / -inf 无穷大/无穷大 ---->NaN
    
// 剔除 nan值=====================================
// NAN != NAN 
    if(!(input[i] == input[i])){
		// input[i] = (Dtype)0.0;   
		// large : -inf  变成负无穷大  就是说input[i]种有  负无穷大  与之前的整体 sigmod激活到 0~1之间 矛盾
	    std::cout<< "======large : " << large << std::endl; // -inf
		//std::cout<< "======id    : " << id << std::endl; //79
		//std::cout<< "=====sum : " << sum << std::endl;// nan
		std::cout<< "======input[10] : " << tep << std::endl; // -inf  input 都变成了 inf
		// 当softmax之前的feature值过大时，由于softmax先求指数，会超出float的数据范围，成为inf
		
		}
  }
  
  return 0;
}

//  类别预测值 分组 类别树yolo9000 激活归一化=========================
template <typename Dtype>
void softmax_tree(Dtype* input, tree *t)
{
  int count = 0;
  for (int i = 0; i < t->groups; ++i)// 遍历每一组
  { //1
    int group_size = t->group_size[i]; // 组内类别数量
    softmax_region(input + count, group_size);// 每组 预测值归一化
    count += group_size; 
  }
}

// 边框预测值 激活处理 处理边框 x,y,w,h 数据 得到预测边框=================================
template <typename Dtype>
vector<Dtype> get_region_box(
   Dtype* x,             // Dtype* x             网络预测输出值
   vector<Dtype> biases, // vector<Dtype> biases 预设边框尺寸权重 
   int n,                // n                    预设边框id  5种 -> (0~4)
   int index,            // index                边框值域 的起始指针id
   int i,                // i , j                当前预测框所属格子 坐标
   int j, 
   int w,                // w , h                总分割格子数量 13 
   int h)
{
  vector<Dtype> b;// 返回的 处理好的 边框数据
  b.clear();
  b.push_back((i + sigmoid(x[index + 0])) / w);// 当前格子偏移量 确定 边框中心点
  b.push_back((j + sigmoid(x[index + 1])) / h);
  b.push_back(exp(x[index + 2]) * biases[2*n] / w);// 考虑 预设框尺寸的 尺寸处理 n ：0~4 5种预设边框
  b.push_back(exp(x[index + 3]) * biases[2*n+1] / h);
  return b;
}

//==============================================================
// 计算 网络输出 边框信息 与标记边框的差分信息==================
//==============================================================
template <typename Dtype>
Dtype delta_region_box(
  vector<Dtype> truth,  // 真实标记的边框  x,y,w,h
  Dtype* x,             // 网络预测输出值x 
  vector<Dtype> biases, // 预设边框尺寸权重 
  int n,                // 预设边框id  5种 -> (0~4)
  int index,            // 边框值域 的起始指针id
  int i,                // i,j 当前预测框所属格子 坐标
  int j, 
  int w,                // w,h 总分割格子数量 13 
  int h, 
  Dtype* delta,         // 输出 网络预测输出值x, 对应的差分信息  delta
  float scale)          // 边框偏差计算尺度权重
  {
  // 获取预测边框=============================================
  vector<Dtype> pred;
  pred.clear();
  // 处理边框 x,y,w,h 数据 得到预测边框=====
  //  get_region_box 会对边框数据进行 激活处理=====
  pred = get_region_box(x, biases, n, index, i, j, w, h);
  
  // 计算 预测边框 和 真实标记边框 的 交并比 用于返回==================
  float iou = Calc_iou(pred, truth);
  
  //LOG(INFO) << pred[0] << "," << pred[1] << "," << pred[2] << "," << pred[3] << ";"<< truth[0] << "," << truth[1] << "," << truth[2] << "," << truth[3];
  // 真实数据反计算 得到 网络应该输出的数据===============
  float tx = truth[0] * w - i; // 0.5
  float ty = truth[1] * h - j; // 0.5
  float tw = log(truth[2] * w / biases[2*n]); // truth[2]= biases/w tw = 0
  float th = log(truth[3] * h / biases[2*n + 1]); // th = 0
  
  
  // 计算偏差信息=============================  dx * x * (1-x)
  delta[index + 0] = (-1) * scale * (tx - sigmoid(x[index + 0])) * sigmoid(x[index + 0]) * (1 - sigmoid(x[index + 0]));
  delta[index + 1] = (-1) * scale * (ty - sigmoid(x[index + 1])) * sigmoid(x[index + 1]) * (1 - sigmoid(x[index + 1]));
  delta[index + 2] = (-1) * scale * (tw - x[index + 2]);
  delta[index + 3] = (-1) * scale * (th - x[index + 3]);
  return iou;
}

//==============================================================
// 计算网络输出 的 类别概率预测信息 的 偏差信息=================
//==============================================================
template <typename Dtype>
void delta_region_class(
Dtype* input_data,  // 网络预测数据 输入
Dtype* &diff,       // 梯度信息
int index,          // 类别预测概率 值域起始指针 id
int class_label,    // 实际类别id 类别标签
int classes,        // 总类别数量  20/80 ......
string softmax_tree,// 类别树文件================yolo9000
tree *t,            // 类别树
float scale,        // 类别概率偏差计算尺度权重
Dtype* avg_cat)     // 对应标签类别的预测概率之和==实际标签 4  预测概率 0.1 0.1 0.05 0.7 ... 该值为0.7
   
{
  if (softmax_tree != "")
   {
		float pred = 1;
		while (class_label >= 0)
		{
			//  对应真实类别的 预测概率===========================
			pred *= input_data[index + class_label];
			
			//LOG(INFO) << "class_label: " << class_label << " p: " << pred; 
			int g = t->group[class_label];// 实际标签=========================
			int offset = t->group_offset[g];
			//LOG(INFO) << "class_label: " << class_label << " p: " << pred << " offset: " << offset; 
			for (int i = 0; i < t->group_size[g]; ++ i)// 遍历 每组
			{
				diff[index + offset + i] = (-1.0) * scale * (0 - input_data[index + offset + i]);
			}
			diff[index + class_label] = (-1.0) * scale * (1 - input_data[index + class_label]);
			class_label = t->parent[class_label];
		}
		
		*avg_cat += pred;//  对应标签类别的预测概率
		//LOG(INFO) << " ";
    } 
   else
    {
		for (int n = 0; n < classes; ++n)// 遍历每一个预测类别概率
		{
			// 对应类别标签值为1，其余类别预测值期望为0， 计算偏差信息
			diff[index + n] = (-1.0) * scale * (((n == class_label) ? 1 : 0) - input_data[index + n]);
			//std::cout<<diff[index+n]<<",";
			if (n == class_label)
			{
				*avg_cat += input_data[index + n];//  对应标签类别的预测概率
				//std::cout<<"avg_cat:"<<input_data[index+n]<<std::endl; 
			}
		}
    }
}

////////////////////////////////////////////////////////
template <typename Dtype>
Dtype get_hierarchy_prob(Dtype* input_data, tree *t, int c)
{
  float p = 1;
  while(c >= 0){
    p = p * input_data[c];
    c = t->parent[c];
  }
  return p;
}

// 层初始化=========================================================
template <typename Dtype>
void RegionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, // 输入
    const vector<Blob<Dtype>*>& top)    // 输出  loss 一维  N*1
{
  LossLayer<Dtype>::LayerSetUp(bottom, top);// 确保 损失层LossLayer的 loss_weight 权重非零
  
  // 获取 RegionLossLayer 的 region_loss_param参数
  RegionLossParameter param = this->layer_param_.region_loss_param();
  
  side_       = param.side();           // 13 最终特征图的尺度   分割的格子数量 13*13
  bias_match_ = param.bias_match();     // 5个边框的10个 biases  边框尺寸 权重参数
  num_class_  = param.num_class();      // 20/80 类别数量
  coords_     = param.coords();         // 4 坐标参数  x,y,w,h
  num_        = param.num();            // 5 每个格子预测5种不同尺寸的 边框
  softmax_    = param.softmax();        // 1 loss 权重
  softmax_tree_ = param.softmax_tree(); // string  无此参数  分类树 yolo9000 会用到
  if (softmax_tree_ != "")
    t_ = tree(softmax_tree_); 
  
  class_map_ = param.class_map();// 无此参数 未传递=============
  if (class_map_ != "")
  {
    string line;
    std::fstream fin(class_map_.c_str());// 读取 类别映射 文件
    if (!fin){
      LOG(INFO) << "no map file";
    }
    
    int index = 0;
    int id = 0;
    while (getline(fin, line)){
      stringstream ss;
      ss << line;
      ss >> id;// 类别id
      
      cls_map_[index] = id;// 类别映射=====================
      index ++;
    }
    fin.close();
  }  

  //LOG(INFO) << "t_.groups: " << t_.groups;
  //jitter_ = param.jitter(); 
  //rescore_ = param.rescore();
  
  object_scale_   = param.object_scale();   //5.0  前景权重
  noobject_scale_ = param.noobject_scale(); //1.0  背景权重
  class_scale_    = param.class_scale();    //1.0  类别权重
  coord_scale_    = param.coord_scale();    //1.0  边框权重
  
  //absolute_ = param.absolute();
  thresh_ = param.thresh(); // 0.5  阈值   边框置信度iou 阈值，大于的话，差分梯度设置为0
  //random_ = param.random();  
  
// 读取5种预设边框 尺寸权重=========================================
  for (int c = 0; c < param.biases_size(); ++c) 
  {
     biases_.push_back(param.biases(c)); 
  } 
  //0.73 0.87;2.42 2.65;4.30 7.04; 10.24 4.59;12.68 11.87;
  //1.08 1.19;3.42 4.41;6.63 11.38;9.42 5.11;16.62 10.52; 
  
// 网络输出   n * ( 5*(类别_nums +4+1) ) * 13 * 13
  int input_count = bottom[0]->count(1); //h*w*n*(classes+coords+1) = 13*13*5*(20+4+1)
// 标签数据   n * (30*5)
  int label_count = bottom[1]->count(1); //30*5 30个边框(4+1)  置信度为1
  
  // outputs: classes, iou, coordinates
  int tmp_input_count = side_ * side_ * num_ * (coords_ + num_class_ + 1); 
  //13*13*5*(20+4+1) label: isobj, class_label, coordinates
  
  int tmp_label_count = 30 * 5; // (class_id + x,y,w,h)
// 预设 30个边框空间，5个为一组边框参数，边框数量不足30， 后面填0
  CHECK_EQ(input_count, tmp_input_count);
  CHECK_EQ(label_count, tmp_label_count);
  
}


template <typename Dtype>
void RegionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) 
{
  LossLayer<Dtype>::Reshape(bottom, top);//定义了 top的形状 
  diff_.ReshapeLike(*bottom[0]);// 输入参数梯度 形状 和 输入形状一直 
  // N *125* 13*13 / N * 425* 13*13  -> N * 169 * 5 * 25 / N * 169 * 5 * 85  
  real_diff_.ReshapeLike(*bottom[0]);// N *125* 13*13 / N * 425* 13*13 
  // diff_ -> real_diff_
  
}

//  层前向传播=======================================
template <typename Dtype>
void RegionLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, // 输入
  const vector<Blob<Dtype>*>& top)    // 输出
{
  //const Dtype* input_data = bottom[0]->cpu_data();// 网络输出 13*13*5*(5+num_class_)
  //std::cout<<"1"<<std::endl;
  const Dtype* label_data = bottom[1]->cpu_data();  // 标签     30 * [label_id,x,y,w,h]
  //std::cout<<"2"<<std::endl;
  
  Dtype* diff = diff_.mutable_cpu_data();// 梯度指针
  caffe_set(diff_.count(), Dtype(0.0), diff);// 初始化梯度为0
  
  //std::cout<<"3"<<std::endl;
  // 初始化 loss输出变量信息
  //      背景(0越好)  物体( 越接近1越好)  交并比(1)                 召回率    损失
  Dtype avg_anyobj(0.0), avg_obj(0.0), avg_iou(0.0), avg_cat(0.0), recall(0.0), loss(0.0);
  int count = 0;
  int class_count = 0;
  
//**************Reshape********************************//
// 网络输出数据变换===============================================
  // 4*125*13*13
  // 4*425*13*13 
  // N*(5*(5+num_class_))*13*13 -> N * (13*13) * 5 * (5+num_class_)
  Blob<Dtype> swap;
  swap.Reshape(bottom[0]->num(), // 图片数量 N
               bottom[0]->height()*bottom[0]->width(), // 特征图尺寸 13*13
               num_,                                   // 一个格子预测5种 边框
               bottom[0]->channels() / num_);          // 5 * (5+num_class_)/5 = (5+num_class_)  25/85 .....
  
  //std::cout<<"4"<<std::endl;  
  // 4+1+20 / 4+1+80
  // 前四个为 边框参数 后面一个为 边框置信度   后面20/80个为 类别概率========================

  Dtype* swap_data = swap.mutable_cpu_data();// cpu上的数据
  caffe_set(swap.count(), Dtype(0.0), swap_data);// 初始化为0
  
  int index = 0;
  for (int b = 0; b < bottom[0]->num(); ++b)// 图片数量 N
    for (int h = 0; h < bottom[0]->height(); ++h)// 13
      for (int w = 0; w < bottom[0]->width(); ++w)// 13
        for (int c = 0; c < bottom[0]->channels(); ++c)// (5*(5+num_class_))
        {
          swap_data[index++] = bottom[0]->data_at(b,c,h,w);// 拷贝数据==============================
        }
    
    //CHECK_EQ(bottom[0]->data_at(0,4,1,2),swap.data_at(0,15,0,4));
    //std::cout<<"5"<<std::endl;
	
    //*******************Activation**************************//
    //disp(swap);
	
/////////////////////////////////////////////////////////////////////////////
 // 1个 边框置信度 sigmoid激活输出===========================================
 //////////////////////////////////////////////////////////////////////////
 
  for (int b = 0; b < swap.num(); ++b)       // 图片数量 N
    for (int c = 0; c < swap.channels(); ++c)// (13*13)  169
      for (int h = 0; h < swap.height(); ++h)// 5
      {//  swap.width() = (5+num_class_) = (4+1+num_class_) = 25/85
        int index = b * swap.channels() * swap.height() * swap.width() + c * swap.height() * swap.width() + h * swap.width() + 4;
        //float tep = swap_data[index];
		// sigmoid 在 
        swap_data[index] = sigmoid(swap_data[index]);
        // 这里 sigmod 激活之后怎么会出现 -nan的值?????  应该是下面出现错误================
        //if(swap_data[index] < 0) std::cout<<"swap_data[index]<0 before sigmod: "<< tep << std::endl;
        CHECK_GE(swap_data[index], 0);// 这里的数据是正常的=================================
      }
        
/////////////////////////////////////////////////////////////////////////
   //std::cout<<"6"<<std::endl;
   
//std::cout<<" softmax_tree_  " << softmax_tree_ << std::endl;
//std::cout<<" num_class_     " << num_class_ << std::endl;

///////////////////////////////////////////////////////////////////////////
// 20/80 个 类别概率预测部分 激活处理======================================
///////////////////////////////////////////////////////////////////////////

  if (softmax_tree_ != "")
  {
    for (int b = 0; b < swap.num(); ++b)// 图片数量 N
      for (int c = 0; c < swap.channels(); ++c)// (13*13)
        for (int h = 0; h < swap.height(); ++h)// 5
        {//  swap.width() = (5+num_class_)
		// 每个边框的类别预测概率 20/80 减去最大值 (-,0]) 在指数映射 -> (0,1] 后归一化
          int index = b * swap.channels() * swap.height() * swap.width() + c * swap.height() * swap.width() + h * swap.width() + 5;
          softmax_tree(swap_data + index, &t_);// 类别树 分组 执行 softmax_region 
        }
  }
  else 
  {
    for (int b = 0; b < swap.num(); ++b)
      for (int c = 0; c < swap.channels(); ++c)
        for (int h = 0; h < swap.height(); ++h)
        {
        // 前5个为 1+4 后面的为 每个边框的类别预测概率 
          int index = b * swap.channels() * swap.height() * swap.width() + c * swap.height() * swap.width() + h * swap.width() + 5;
        // 类别概率  减去最大值 (-,0]) 在指数映射 -> (0,1] 后归一化
          softmax_region(swap_data+index, num_class_);// 20/80类
        // 这里 sigmod 激活之后怎么会出现 -nan的值?????
        //if(swap_data[index] < 0) std::cout<<"swap_data[index]<0 before sigmod: "<< tep << std::endl;
        for (int i = 0; i < num_class_; ++i)
            CHECK_GE(swap_data[index + i], 0);
        }
  }
    //std::cout<<"7"<<std::endl;
    //disp(swap);
    //LOG(INFO) << "data ok!";
    //*********************************************************Diff********************************************************//
  int best_num = 0;
  for (int b = 0; b < swap.num(); ++b){// 图片数量 N
  
    if (softmax_tree_ != ""){
      int onlyclass = 0;
	  
// 分类 loss 求梯度======================================================
      for (int t = 0; t < 30; ++t){// 30个 标签边框   class_label,x,y,w,h
// 取出标签数据=============================================
        vector<Dtype> truth;
        Dtype x = label_data[b * 30 * 5 + t * 5 + 1];// 中心点
        Dtype y = label_data[b * 30 * 5 + t * 5 + 2];
        if (!x) break;

        int class_label = label_data[b * 30 * 5 + t * 5 + 0];// 标记值 物体类别
        float maxp = 0;
        int maxi = 0;
        if (x > 100000 && y > 100000)// 特殊值，标记为只有 类别，无边框数据
		{
          for (int j = 0; j < side_; ++ j)// 13
            for (int i = 0; i < side_; ++ i)// 13
			
              for (int n = 0; n < num_; ++n)
			  { // 5 种预测边框
                int index = b * swap.channels() * swap.height() * swap.width() + (j * side_ + i) * swap.height() * swap.width() + n * swap.width();
                //  swap_data : (13*13) * 5 * (5+num_class_)
                float scale = swap_data[index + 4];// 边框置信度
				
 // 边框置信度 梯度计算========================================
                diff[index + 4] = (-1.0) * noobject_scale_ * (0 - swap_data[index + 4]) * (swap_data[index + 4]) * (1 - swap_data[index + 4]);
                float p = scale * get_hierarchy_prob(swap_data + index + 5, &t_, class_label);
                if (p > maxp)
				{
                  maxp = p;
                  maxi = index;// 最大 概率
                }
              }
          //LOG(INFO) << "delta hierarchy prob";
// 类别概率梯度=====================
          delta_region_class(swap_data, diff, maxi + 5, class_label, num_class_, softmax_tree_, &t_, class_scale_, &avg_cat);
          if (swap_data[maxi + 4] < 0.3) 
			  // 记录 类别 概率 大于 0.3 的梯度===================================
			  diff[maxi + 4] = -1 * object_scale_ *  (0.3 - swap_data[maxi + 4]) * (swap_data[maxi + 4]) * (1 - swap_data[maxi + 4]);
          else 
			  diff[maxi + 4] = 0;
          ++ class_count;
          onlyclass = 1;
          break;
        }
      }
      //LOG(INFO)<< "tree ok";
      if (onlyclass) continue;
    }
	
// 边框 loss 梯度=============================================================
    for (int j = 0; j < side_; ++j)// 13
      for (int i = 0; i < side_; ++i)// 13
        for (int n = 0; n < num_; ++n){// 5
		//  n* 169  * 5 * 25 / n * 169  * 5 * 85 
		//                   169                   5          25/85
          int index = b * swap.channels() * swap.height() * swap.width() + (j * side_ + i) * swap.height() * swap.width() + n * swap.width();
          CHECK_EQ(swap_data[index], swap.data_at(b, j * side_ + i, n, 0));
          //std::cout<<index<<std::endl;
		  
//  从网络获取 边框预测值 get_region_box 会对边框数据进行 激活处理=====================
          vector<Dtype> pred = get_region_box(swap_data, biases_, n, index, i, j, side_, side_);
          float best_iou = 0;
		  
          for (int t = 0; t < 30; ++t)// 30个 标签边框   class_label,x,y,w,h
		  {
		  // 获取所有的真实 标记边框===================================
            vector<Dtype> truth;
			// 标记值边框数据==============================
            Dtype x = label_data[b * 30 * 5 + t * 5 + 1];
            Dtype y = label_data[b * 30 * 5 + t * 5 + 2];
            Dtype w = label_data[b * 30 * 5 + t * 5 + 3];
            Dtype h = label_data[b * 30 * 5 + t * 5 + 4];

            if (!x) break;
            truth.push_back(x);
            truth.push_back(y);
            truth.push_back(w);
            truth.push_back(h);
			// 预测和标签 计算 交并比==============================
            Dtype iou = Calc_iou(pred, truth);
            if (iou > best_iou) best_iou = iou;// 记录最好的，也就是该预测边框所预测的 目标边框=====
			/////// 打印调试信息  查看输入标签数据是否正确===============================
         // if (i + j + n == 0)
          //  LOG(INFO)<<"label,x,y,w,h=["<< label_data[t * 5 + 0] << ","<< x << " " << y << " " << w << " " << h <<"]";
          }
          //std::cout<<"anyobj:"<<swap_data[index+4];
		  // 
		  
// 记录预测边框 和目标标记边框的 差分 梯度=====================================
          avg_anyobj += swap_data[index + 4];// 边框置信度
          diff[index + 4] = -1 * noobject_scale_ * (0 - swap_data[index + 4]) * (swap_data[index + 4]) * (1 - swap_data[index + 4]);
          
		  /////////////////////////////////////////////////////////////////////////
		 // std::cout<<"diff:"<<diff[index+4]<<std::endl;
		  
		  
          if (best_iou > thresh_)
		  {
            best_num ++;
            diff[index + 4] = 0;// 预测的很准确 梯度为0
            // LOG(INFO)<<"best_iou: "<<best_iou<<" index:"<<index;
          }
// 最开始的12800次迭代 还要计算 预测边框 和 预设边框的 差分梯度===================================
          if (iter < 12800 / bottom[0]->num())
		  { // 12800/图片数量 N
            vector<Dtype> truth;
            truth.clear();
            truth.push_back((i + .5) / side_); // center of i,j
            truth.push_back((j + .5) / side_); // 坐标中心 得到格子偏移
            truth.push_back((biases_[2 * n]) / side_); // 预设框尺寸 anchor boxes
            truth.push_back((biases_[2 * n + 1]) / side_);
            delta_region_box(truth, swap_data, biases_, n, index, i, j, side_, side_, diff, .01);
          }
        }
    //std::cout<<"2####"<<index<<std::endl;    
    //std::cout<<"best_num:"<<best_num<<std::endl;
    //LOG(INFO) << "obj ok";
	
    for (int t = 0; t < 30; ++t)
	{ // 最多 30个 标签边框   class_label,x,y,w,h
      vector<Dtype> truth;
      truth.clear();
      // 真实标记数据================================================
      int class_label = label_data[t * 5 + b * 30 * 5 + 0];
      float x = label_data[t * 5 + b * 30 * 5 + 1];
      float y = label_data[t * 5 + b * 30 * 5 + 2];
      float w = label_data[t * 5 + b * 30 * 5 + 3];
      float h = label_data[t * 5 + b * 30 * 5 + 4];
            //LOG(INFO) << x << " " << y << " " << w << " " << h;
      if (!w) break;
      truth.push_back(x);
      truth.push_back(y);
      truth.push_back(w);
      truth.push_back(h);
      float best_iou = 0;
      int best_index = 0;
      int best_n = 0;
      int i = truth[0] * side_; // 实际格子  match which i,j
      int j = truth[1] * side_;
      int pos = j * side_ + i;// 
        
      vector<Dtype> truth_shift;
      truth_shift.clear();
      truth_shift.push_back(0);
      truth_shift.push_back(0);
      truth_shift.push_back(w);
      truth_shift.push_back(h);
      //std::cout<<"3####"<<std::endl;
      int size = coords_ + num_class_ + 1; //4 + 20 + 1 = 25   /  85...

      for (int n = 0; n < num_; ++ n){ // 5种预测边框 search 5 anchor in i,j
        int index = b * bottom[0]->count(1) + pos * size * num_ + n * size;
		
//  从网络获取 边框预测值 get_region_box 会对边框数据进行 激活处理=====================                         
        //std::cout<<"#########1"<<index<<std::endl;
        //int index = 25 * (j * side_ * 5 + i * 5 + n) + b * bottom[0]->count(); //25 * (j * 13 * 5 + i * 5 + n)
        vector<Dtype> pred = get_region_box(swap_data, biases_, n, index, i, j, side_, side_); //
        
		if (bias_match_)
		{
          pred[2] = biases_[2 * n] / side_;
          pred[3] = biases_[2 * n + 1] / side_;
        }
        //std::cout<<"#########2"<<index<<std::endl;
        pred[0] = 0;
        pred[1] = 0;
        float iou = Calc_iou(pred, truth_shift);
        if (iou > best_iou){
          best_index = index;
          best_iou = iou;
          best_n = n;
        }
        //std::cout<<"#########3"<<index<<std::endl;
      }
      //LOG(INFO) << "iou ok";
          //std::cout<<"best_n:"<<best_n<<" best_iou:"<<best_iou<<std::endl;
      //std::cout<<"4####"<<std::endl;
      //LOG(INFO)<<"2_t:"<<t<<" best_iou:"<<best_iou;
      float iou = delta_region_box(truth, swap_data, biases_, best_n, best_index, i, j, side_, side_, diff, coord_scale_);
      //LOG(INFO) << "iou:" << iou;	
      if (iou > 0.5) recall += 1;
      avg_iou += iou;
      //std::cout<<"obj"<<swap_data[best_index+4]<<std::endl;
      avg_obj += swap_data[best_index + 4];
          //LOG(INFO)<<"avg_obj:"<<avg_obj;
      diff[best_index + 4] = (-1.0) * object_scale_ * (1 - swap_data[best_index + 4]) * (swap_data[best_index + 4] * (1 - swap_data[best_index + 4]));
      
	  //if (rescore)
///////////////////////////////////////////////////////////////////// 
      //std::cout<<"diff:"<<diff[best_index+4]<<std::endl;
      //LOG(INFO) << best_index << " " << best_n;

      //if (l.map) class_label = l.map;
      //LOG(INFO) << "cls_label: " << class_label;
          
      if (class_map_ != "") class_label = cls_map_[class_label];
      delta_region_class(swap_data, diff, best_index + 5, class_label, num_class_, softmax_tree_, &t_, class_scale_, &avg_cat); //softmax_tree_
/////////////////////////////////////////////////////////////////////////////////
      //std::cout<<"###################real diff#################"<<std::endl;
      //for (int i = 0; i < num_class_; i ++)
      //std::cout<<diff[best_index+5+i]<<",";
      //std::cout<<std::endl;
      //LOG(INFO) << "class ok";
////////////////////////////////////////
      ++count;
      ++class_count;
    }
     // std::cout<<"5####"<<std::endl;
  }
  
  //std::cout<<"8"<<std::endl;
  //caffe_set(diff_.count(), Dtype(0.0), diff);
  
  // N * 13*13 * 5 * (4 + 1 + num_class_)
  //                        N            13*13=169                          5        25/85
  diff_.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(), num_, bottom[0]->channels() / num_);
  //disp(diff_);

  Dtype* real_diff = real_diff_.mutable_cpu_data();    
  //std::cout<<"9"<<std::endl;
  int sindex = 0;

  for (int b = 0; b < real_diff_.num(); ++b)// N
    for (int h = 0; h < real_diff_.height(); ++h)// 13
      for (int w = 0; w < real_diff_.width(); ++w)// 13
        for (int c = 0; c < real_diff_.channels(); ++c)// 125 / 425 
        {
          int rindex = b * real_diff_.height() * real_diff_.width() * real_diff_.channels() + c * real_diff_.height() * real_diff_.width() + h * real_diff_.width() + w;
          Dtype e = diff[sindex];
          //Dtype e = 1;
          //std::cout<<"index: "<< rindex <<" sindex: "<<sindex<<", "<<e<<std::endl;
          real_diff[rindex] = e;
          sindex++;
        }

  //std::cout<<"10"<<std::endl;
  //disp(real_diff_);
  //LOG(INFO) << avg_anyobj;	
  // LOG(INFO) << swap.shape_string();
  //LOG(INFO) << bottom[1]->shape_string();
  // LOG(INFO) << bottom[0]->count();
  
//  计算 损失 =======================================================
  for (int i = 0; i < real_diff_.count(); ++i)
  {
    loss += real_diff[i] * real_diff[i];
  }
// 输出损失==================================
  top[0]->mutable_cpu_data()[0] = loss;
  
  //LOG(INFO) << "avg_noobj: " << avg_anyobj / (side_ * side_ * num_ * bottom[0]->num());	
  iter ++;
  //LOG(INFO) << "iter: " << iter <<" loss: " << loss;
  if (!(iter % 100))
  {
    LOG(INFO) << "avg_noobj: "<< avg_anyobj/(side_*side_*num_*bottom[0]->num()) << " avg_obj: " << avg_obj/count <<" avg_iou: " << avg_iou/count << " avg_cat: " << avg_cat/class_count << " recall: " << recall/count << " class_count: "<< class_count;
  }
}
////////////////////////////////// 反向传播//////////////////////////////////
template <typename Dtype>
void RegionLossLayer<Dtype>::Backward_cpu(
const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, // 是否需要更新
const vector<Blob<Dtype>*>& bottom) 
{
  //LOG(INFO) <<" propagate_down: "<< propagate_down[1] << " " << propagate_down[0];
  if (propagate_down[1]) 
  {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  
  if (propagate_down[0]) 
  {
	  
    const Dtype sign(1.);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();// 0.125
	
/////////////////////test////////////
    //const Dtype alpha(1.0);
    //LOG(INFO) << "alpha:" << alpha;
    
    caffe_cpu_axpby(
        bottom[0]->count(),
        alpha,
        real_diff_.cpu_data(),
        Dtype(0),
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
//STUB_GPU(DetectionLossLayer);
#endif

INSTANTIATE_CLASS(RegionLossLayer);
REGISTER_LAYER_CLASS(RegionLoss);

}  // namespace caffe
