/////////////////////// yolov2检测评估层 //////

#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

//#include "caffe/layers/detection_loss_layer.hpp"
#include "caffe/layers/eval_detection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

class BoxData {
 public:
  int label_;
  float score_;
  vector<float> box_;
};

inline float sigmoid(float x)
{
  return 1. / (1. + exp(-x));
}

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

// 类别概率激活===================================== 
// 类别概率部分 归一化处理
// 类别概率  减去最大值 (-,0]) 在指数映射 -> (0,1] 后归一化
template <typename Dtype>
float softmax_region(Dtype* input, int classes)
{
  Dtype sum = 0;
  Dtype large = input[0];// 初始化最大值
  
  // 遍历记录 类别预测概率 最大值
  for (int i = 0; i < classes; ++i){
    if (input[i] > large)
      large = input[i];// 类别概率最大值
  }
// 减最大值 (-inf,0]) 在指数映射 -> (0,1] ，后求和
  for (int i = 0; i < classes; ++i){
    Dtype e = exp(input[i] - large);
    sum += e;// 求和
    input[i] = e;
  }
  for (int i = 0; i < classes; ++i){
    input[i] = input[i] / sum;// 归一化 
  }
  return 0;
}

// 排序 标准 按照得分box1.score_ 排序
bool BoxSortDecendScore(
const BoxData& box1, 
const BoxData& box2) 
{
  return box1.score_ > box2.score_;
}

// NMS 非极大值抑制 剔除重叠度较高的 边框 返回留下的边框id
void ApplyNms(const vector<BoxData>& boxes, vector<int>* idxes, float threshold) {
  map<int, int> idx_map;
  for (int i = 0; i < boxes.size() - 1; ++i) {
    
    if (idx_map.find(i) != idx_map.end()) {
      continue;// 该边框已经被剔除
    }
	
    vector<float> box1 = boxes[i].box_;// 最高得分的边框
	
    for (int j = i + 1; j < boxes.size(); ++j) {// 与剩下的 进行重叠度计算
	
      if (idx_map.find(j) != idx_map.end()) {
        continue;// 该边框已经被剔除
      }
	  
      vector<float> box2 = boxes[j].box_;
      float iou = Calc_iou(box1, box2);
      if (iou >= threshold) // 与最高得分的边框 重叠度较大
	  {
        idx_map[j] = 1;// 需要剔除
      }
    }
  }
  for (int i = 0; i < boxes.size(); ++i) {
    if (idx_map.find(i) == idx_map.end()) {
      idxes->push_back(i);
    }
  }
}

// 返回一张图像的 标签 + 多边框数据的 标签
template <typename Dtype>
void GetGTBox(int side, 
              const Dtype* 
			  label_data, 
			  map<int, vector<BoxData> >* gt_boxes) 
{
  // 输入数据指针数组  label_data 以及确定对应图片的起始地址
  // 数据形式  30*5  一张图片 30个物体边框 第一个为标签 后面的为 边框 x y w h
  //int locations = pow(side, 2);
  //for (int i = 0; i < locations; ++i) {
  for (int i = 0; i < 30; ++ i) {    // 30个物体边框
    //if (!label_data[locations + i]) {
    if (label_data[i * 5 + 1] == 0) 
	{// 有的图片标注框没有30个，其余设置为0
	 //continue;
	 break; //开始出现0了，后面应该全部是0 跳过全部
    }
    BoxData gt_box;
    //bool difficult = (label_data[i] == 1);
    //int label = static_cast<int>(label_data[locations * 2 + i]);
    int label = label_data[i * 5 + 0];// 标签
    //gt_box.difficult_ = difficult;
    gt_box.label_ = label;
	
//////////////////////////////////////////////////////////////////////
///////////////////////////////////////////
    // gt_box.score_ = 1;
    gt_box.score_ = i; // 真实边框 的得分应该设置为1  
    //int box_index = locations * 3 + i * 4;
    int box_index = i * 5 + 1;// 边框起始指针 x y w h
    //LOG(INFO) << "label:" << label;
    for (int j = 0; j < 4; ++j) {
      gt_box.box_.push_back(label_data[box_index + j]);
	  
 // 打印调试信息  查看输入标签数据是否正确===============================
      //LOG(INFO) << "x,y,w,h:" << label_data[box_index + j];
    }
    if (gt_boxes->find(label) == gt_boxes->end()) 
	{
      (*gt_boxes)[label] = vector<BoxData>(1, gt_box);// 一类中唯一一个物体
    } 
	else {
      (*gt_boxes)[label].push_back(gt_box);// 一类中多个物体
    }
  }
}

// 获取预测边框 13*13*5 NMS抑制+置信度抑制==================================
template <typename Dtype>
void GetPredBox(int side, int num_object,  //  格子 13  5个物体  20/80 种类别
               int num_class, Dtype* input_data,  // 输入数据 13*13*125/13*13*425 -> 13*13*5*25/13*13*5*85
               map<int, vector<BoxData> >* pred_boxes, //  nms之后输出的边框   评分类别  nms阈值(重叠)
  int score_type, float nms_threshold, vector<Dtype> biases) {

 vector<BoxData> tmp_boxes;// 筛选出来的  13*13*5个格子
  //int locations = pow(side, 2);
  for (int j = 0; j < side; ++j)// 13    0->12 格子
    for (int i = 0; i < side; ++i)// 13  0->12 
      for (int n = 0; n < 5; ++n)// 5    0->4  物体边框数量
      {// 25 = 4边框参数 + 1置信度 + 20种类别概率  
	  
	// 起始指针
	int index = (j * side + i) * num_object * (num_class + 1 + 4) + n * (num_class + 1 + 4);
	
// 坐标中心 sigmoid激活得到格子偏移量===========================================
	float x = (i + sigmoid(input_data[index + 0])) / side;// 格子偏移量/总格子数量 归一化到 0~1之间
	float y = (j + sigmoid(input_data[index + 1])) / side;
	// 边框长度指数映射之后 分别被五种边框尺寸系数加权之后 除以/总格子数量 归一化到 0~1之间
 	float w = (exp(input_data[index + 2]) * biases[2 * n]) / side;
	float h = (exp(input_data[index + 3]) * biases[2 * n + 1]) / side;

    // 置信度 在后面处理 需要 sigmoid() 处理到0~1==============================
    // 从20种预测概率种选出概率最大的，作为本边框的预测 类比===================
// 20种 类别预测概率处理=======================================================
	softmax_region(input_data + index + 5, num_class);
	int pred_label = 0;
	
// 在20种 类别预测概率 种选出 概率最大的 ======================================
	float max_prob = input_data[index + 5];// 初始化第一种物体的预测概率为最大
	
	for (int c = 0; c < num_class; ++c)
	{
	  if (max_prob < input_data[index + 5 + c])
	  {
	    max_prob = input_data[index + 5 + c];// 记录预测概率 最大的值
	    pred_label = c; // 0,..,19(20类)     对应的类别 标签
	  }
	}
	BoxData pred_box;
	pred_box.label_ = pred_label;// 预测类别标签

	
 // 置信度 需要 sigmoid() 处理到0~1===========================================
    float obj_score = sigmoid(input_data[index + 4]);
	
// 预测得分====================================================================
	if (score_type == 0) 
	{
	  pred_box.score_ = obj_score;// 按照 置信度 进行评价
	} 
	else if (score_type == 1) 
	{
	  pred_box.score_ = max_prob; // 按照 类别预测概率进行评价
	} 
	else 
	{
	  pred_box.score_ = obj_score * max_prob;// 按照 置信度*类别预测概率 进行评价
	}
// 预测边框===============================
	pred_box.box_.push_back(x);
	pred_box.box_.push_back(y);
   	pred_box.box_.push_back(w);
	pred_box.box_.push_back(h);
	
	tmp_boxes.push_back(pred_box);// 13*13*5个格子
	//LOG(INFO)<<"Not nms pred_box:" << pred_box.label_ << " " << obj_score << " " << max_prob  << " " << pred_box.score_ << " " << pred_box.box_[0] << " " << pred_box.box_[1] << " " << pred_box.box_[2] << " " << pred_box.box_[3];	
    }  
  
  //  进行NMU非极大值抑制 滤出 重叠度较高的框
  if (nms_threshold >= 0) 
  {
	  
    // 排序 标准 按照得分box1.score_ 排序
    std::sort(tmp_boxes.begin(), tmp_boxes.end(), BoxSortDecendScore);
    vector<int> idxes;
	// NMS 非极大值抑制 剔除重叠度较高的 边框 返回留下的边框id
    ApplyNms(tmp_boxes, &idxes, nms_threshold);
	
    for (int i = 0; i < idxes.size(); ++i) {
      BoxData box_data = tmp_boxes[idxes[i]];
      //**************************************************************************************//
      if (box_data.score_ < 0.005) // 得分较小 跳过 from darknet
          continue;
      //LOG(INFO)<<"box_data:" << box_data.label_ << " " << box_data.score_ << " " << box_data.box_[0] << " " << box_data.box_[1] << " " << box_data.box_[2] << " " << box_data.box_[3];
      if (pred_boxes->find(box_data.label_) == pred_boxes->end()) 
	  {
        // 一种类别 单个物体边框
        (*pred_boxes)[box_data.label_] = vector<BoxData>();
      }  // 多种同类型物体 多个人
      (*pred_boxes)[box_data.label_].push_back(box_data);
    }
  } 
  else 
  {
    for (std::map<int, vector<BoxData> >::iterator it = pred_boxes->begin(); it != pred_boxes->end(); ++it) 
	{
      std::sort(it->second.begin(), it->second.end(), BoxSortDecendScore);
    }
  }
}

// 层 初始化======================================================================
template <typename Dtype>
void EvalDetectionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, // 输入
    const vector<Blob<Dtype>*>& top)    // 输出
{
  EvalDetectionParameter param = this->layer_param_.eval_detection_param();
  side_ = param.side();            // 13*13格子
  num_class_ = param.num_class();  // voc 20类 / coco 80类
  num_object_ = param.num_object();// 一个格子预测5个边框
  threshold_ = param.threshold();  // 预测正确/错误 IOU阈值
  //sqrt_ = param.sqrt();
  //constriant_ = param.constriant();
  
  nms_ = param.nms();// nms 阈值
  
  for (int c = 0; c < param.biases_size(); ++c)
  {  // 5种边框尺寸 10个参数
    biases_.push_back(param.biases(c));// 网络输出 解码 到 预测边框尺寸需要用到
  }

  switch (param.score_type()) 
  {
    case EvalDetectionParameter_ScoreType_OBJ:
      score_type_ = 0;
      break;
    case EvalDetectionParameter_ScoreType_PROB:
      score_type_ = 1;
      break;
    case EvalDetectionParameter_ScoreType_MULTIPLY:
      score_type_ = 2;
      break;
    default:
      LOG(FATAL) << "Unknow score type.";
  }
}

template <typename Dtype>
void EvalDetectionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) 
{
  int input_count = bottom[0]->count(1); //13*13*125 / 13*13*425 网络输出
  int label_count = bottom[1]->count(1); //30*5      标签 预设30个物体 4边框+1类别
  // outputs: classes, iou, coordinates
  //int tmp_input_count = side_ * side_ * (num_class_ + (1 + 4) * num_object_);
  int tmp_input_count = side_ * side_ * num_object_ *( num_class_ + 4 + 1 ); //13*13*5*25
  // label: isobj, class_label, coordinates
  //int tmp_label_count = side_ * side_ * (1 + 1 + 1 + 4);
  int tmp_label_count = 30 * 5;// 

 // LOG(INFO) << " label[0] : " << bottom[1]->count(0) << "================================";
  
  
  CHECK_EQ(input_count, tmp_input_count);// 确保网络输出 每张图片为 13*13*5*25 大小 
                                         // 13*13个格子，每个格子预测5种边框,每种边框预测 20类概率+4边框参数+1置信度
  CHECK_EQ(label_count, tmp_label_count);// 而标签输入   每张图片为 30*5       大小 30个物体边框，4个边框参数+1个类别标签

   vector<int> top_shape(2, 1);// 两行一列 全1
  //vector<int> top_shape(3, 1);// 三行一列  添加一列 存储mAP
  top_shape[0] = bottom[0]->num();// 图片数量
  //top_shape[1] = num_class_ + side_ * side_ * num_object_ * 4; 
  // 20 + 13*13*5*4 标签 得分 TP FP 
 // top_shape[1] = num_class_ + side_ * side_ * num_object_ * 4 + 1; 
  // 各个类别预测数量,  20类检测数量 + 13*13*5*(label + score + tp + fp) + 该图片mAP
  
   top_shape[1] = side_ * side_ * num_object_ * 4 + 1; 
  // 13*13*5*(label + score + tp + fp) + 该图片mAP
  
  //top_shape[2] = 1;// 添加一列 存储mAP
  top[0]->Reshape(top_shape);

}

template <typename Dtype>
void EvalDetectionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  //const Dtype* input_data = bottom[0]->cpu_data();// 网络输出     N*13*13*125
  const Dtype* label_data = bottom[1]->cpu_data();  // 真实标签数据 N*30*5
  //LOG(INFO) << bottom[0]->data_at(0,0,0,0) << " " << bottom[0]->data_at(0,0,0,1);  
  
  Blob<Dtype> swap;// 网络输出数据 N * 13* 13* 125
  // 变形为    N * (13 * 13) *  5 * 25
 // N*(5*(5+num_class_))*13*13 -> N * (13*13) * 5 * (5+num_class_)
  swap.Reshape(bottom[0]->num(), 
               bottom[0]->height()*bottom[0]->width(), 
               num_object_, 
               bottom[0]->channels()/num_object_);  
  
  Dtype* swap_data = swap.mutable_cpu_data();// cpu上的数据
  caffe_set(swap.count(), Dtype(0.0), swap_data);// 设置为0
  int index = 0;
  for (int b = 0; b < bottom[0]->num(); ++b)// 图片数量
    for (int h = 0; h < bottom[0]->height(); ++h) // 格子 13
      for (int w = 0; w < bottom[0]->width(); ++w)// 格子 13
        for (int c = 0; c < bottom[0]->channels(); ++c)// 5*25=125
	{
	  swap_data[index++] = bottom[0]->data_at(b,c,h,w);
	}  
  //*******************************************************************************//
  //caffe_set(swap.count(), Dtype(0.0), swap_data);// 设置为0
  //int p_index = (7*13+4)*125;
  //swap_data[p_index]=-0.1020;
  //swap_data[p_index+1]=2.0867;
  //swap_data[p_index+2]=1.612;
  //swap_data[p_index+3]=1.0515;
  //swap_data[p_index+4]=1.0;
  //swap_data[p_index+5+11]=100;  

  //*******************************************************************************//  
  Dtype* top_data = top[0]->mutable_cpu_data();// 层输出 cpu数据
  caffe_set(top[0]->count(), Dtype(0), top_data);// 设置为0
  Dtype all_mAP = 0.0;// 总 batch 的 mAP
  for (int i = 0; i < bottom[0]->num(); ++i) 
 {   // N  图片数量
    int input_index = i * bottom[0]->count(1);// 网络输出标签 i * 13*13*125
    int true_index = i * bottom[1]->count(1);//  真实标签     i * 30*5
    int top_index = i * top[0]->count(1);    //  输出数据    //  i * ( 20 + 13*13*5*4 + 1) -> i * (13*13*5*4 + 1)
                                             //  前面20个为 真实标签 物体类别出现的次数
 
 // 获取真实边框 =========================================
    map<int, vector<BoxData> > gt_boxes;
    // 从 对应图片的标签数据中 获取 真实边框 label_ + score_ + box_ 
    // 返回一张图像的 标签 + 多边框数据的 标签
    GetGTBox(side_, label_data + true_index, &gt_boxes);

 
// 在输出数据中  记录 真实标签 物体类别出现的次数=======================
    for (std::map<int, vector<BoxData > >::iterator it = gt_boxes.begin(); it != gt_boxes.end(); ++it) 
	{
      // 遍历 每一个 真实的标签
      //int label = it->first;// 标签 类别
      vector<BoxData>& g_boxes = it->second;// BoxData: label_ + score_ + box_ 
      for (int j = 0; j < g_boxes.size(); ++j) 
	  {// 边框数量
         // 输出数据中的前 20个================================================
// =======================================
         // top_data[top_index + label] += 1;     // 真实标签 物体类别出现的次数
      }
    }
// 获取预测边框 =============================================
    map<int, vector<BoxData> > pred_boxes;
    // 获取预测边框 13*13*5 -> NMS抑制+置信度抑制 ->  pred_boxes(数量少很多)
    //GetPredBox(side_, num_object_, num_class_, input_data + input_index, &pred_boxes, sqrt_, constriant_, score_type_, nms_);
    GetPredBox(side_, num_object_, num_class_, swap_data + input_index, &pred_boxes, score_type_, nms_, biases_);
    
	// 输出数据 后面 的 13*13*5*4 =============================
    // int index = top_index + num_class_ + 1;// 20 + 1 之后为 上面的 (label + score + tp + fp) 参数
//=============================
    int index = top_index + 1;// 1个 map 之后为 上面的 (label + score + tp + fp) 参数
	
    int pred_count(0);// 
    
    Dtype mAP = 0.0;
	int pre_clas_num=0;
	//float AP = 0.0;
	// int tp
    // 遍历预测值的 每一类，与最合适的标签边框计算AP,在计算总类别的mAP============
    for (std::map<int, vector<BoxData> >::iterator it = pred_boxes.begin(); it != pred_boxes.end(); ++it) 
	{
      Dtype AP = 0.0;// 该类AP
	  int tp=0;// 预测正确
	  int fp=0;// 预测错误
	  ++pre_clas_num;// 该图片预测的总类别数量
      int label = it->first;// 预测 边框标签
      vector<BoxData>& p_boxes = it->second;// 多个边框 vector<BoxData>
	  
// 真实标签中 未找到该 类别=======================================
      if (gt_boxes.find(label) == gt_boxes.end()) {
        for (int b = 0; b < p_boxes.size(); ++b) {// 该类别下的每一个预测边框
          top_data[index + pred_count * 4 + 0] = p_boxes[b].label_;// 标签
          top_data[index + pred_count * 4 + 1] = p_boxes[b].score_;// 得分
          top_data[index + pred_count * 4 + 2] = 0; //tp 
          top_data[index + pred_count * 4 + 3] = 1; //fp 错误的预测为正确的 
          ++pred_count;
          ++fp;// 预测错误============================================
        }
		if(tp + fp)  
           AP = tp / (tp + fp);// 计算该类别的 平均准确度  这里等于0
		mAP += AP;// 所有类别的总 AP  这里可以省略 因为 AP为0
        continue;// 跳过预测错误的，只记录fp
      } 
  // 真实标签找到了该预测的类别======================================
      vector<BoxData>& g_boxes = gt_boxes[label];// 真实标签中该 类别的 多个真实边框=====
	  
      vector<bool> records(g_boxes.size(), false);// 记录 真实类别的每一个 物体边框 是否已经被预测过
	  
      for (int k = 0; k < p_boxes.size(); ++k) 
	  { // 遍历 每个 预测边框==============
        top_data[index + pred_count * 4 + 0] = p_boxes[k].label_;// 标签
        top_data[index + pred_count * 4 + 1] = p_boxes[k].score_;// 得分
        Dtype max_iou(-1);// 预测边框最接近的  真实边框 iou
        int idx(-1);// 对应的 真实边框id
		
		// 遍历每个真实边框 找到 预测边框最接近的 真实边框===========
        for (int g = 0; g < g_boxes.size(); ++g) {
          Dtype iou = Calc_iou(p_boxes[k].box_, g_boxes[g].box_);// 计算交并比
          if (iou > max_iou) {
            max_iou = iou;// 记录 每个预测边框 最接近的  真实边框 的 IOU
            idx = g;// 对应的 真实边框id
          }
        }
		// 根据 交并比 确定判断 预测 正确/错误
        if (max_iou >= threshold_) { 
             
            if ( !records[idx] ) {
              records[idx] = true;// 对应的 真实边框id
              top_data[index + pred_count * 4 + 2] = 1; // tp  正->正确
              top_data[index + pred_count * 4 + 3] = 0; // fp
              ++tp;// 预测正确 =================================
            } 
            else 
            {// 同一个位置的物体之前已经被预测过，又一个框来预测，则认为是错误的
              top_data[index + pred_count * 4 + 2] = 0;
              top_data[index + pred_count * 4 + 3] = 1;// 错误 -> 正确
              ++fp;// 预测错误============================================
            }
        }
        ++pred_count;
      }
      if(tp + fp)  AP = tp / (tp + fp);// 计算该类别的 平均准确度
      mAP += AP;// 所有类别的总 AP  
    }
	if(pre_clas_num){
		mAP /= pre_clas_num;
		// mAP /= num_class_; // 计算 mAP 是除以总类别数，还是总预测类别数
	} 
	else mAP = 0.0;
    // 输出对应图片的mAP
    top_data[ index - 1 ] = mAP;
	all_mAP += mAP;
  }
  if(bottom[0]->num()) all_mAP /= (bottom[0]->num());
  top_data[0] = all_mAP;
}

INSTANTIATE_CLASS(EvalDetectionLayer);
REGISTER_LAYER_CLASS(EvalDetection);

}  // namespace caffe
