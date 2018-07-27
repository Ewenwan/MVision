// 标注数据层==================================================
// 转换检测数据AnnotatedDatum=================
// 图像数据
// datum = anno_datum->mutable_datum()
//  datum->set_data();                      // 所有8位像素数据
//  datum->set_channels(cv_img.channels()); // 通道数量
//  datum->set_height(cv_img.rows);         // 行
//  datum->set_width(cv_img.cols);          // 列
//  datum->set_encoded(true);               // 编码?
// 标签
// anno_datum->mutable_annotation_group(g)->set_group_label(label)   标签
//                                        ->mutable_bbox()->set_xmin();  边框
//                                                        ->set_ymin();
//                                                        ->set_xmax();
//                                                        ->set_ymax();
//                                                        ->set_difficult();
//======================================================
////////////////////////////////////////////////////////////////////////////////
//  读取数据库文件到队列，从队列中取数据，生成处理后的 批数据 Batch
// batch->data_                      为多个图像数据域
// batch->label_.mutable_cpu_data(); 为标签  每8个为1个边框标签
// top_label[idx++] = item_id;// batch id 0~3  / 0~8 / 0~15 图片id
// top_label[idx++] = anno_group.group_label();// 类别标签 id
// top_label[idx++] = anno.instance_id();      // 物体个数 实例id     
// top_label[idx++] = bbox.xmin();             // 左上角  
// top_label[idx++] = bbox.ymin();
// top_label[idx++] = bbox.xmax();             // 右下角
// top_label[idx++] = bbox.ymax();
// top_label[idx++] = bbox.difficult();        // ? 数据难度??
//////////////////////////////////////////////////////////////////////
////// ssd 格式
///////////////////////////
// 输出 top的形式=====================================
// top[0] 为 数据区 data 
// top_data = batch->data_.mutable_cpu_data();
// top[0]->num()       图片数量 N
// top[0]->channels()  通道数量 3
// top[0]->height()    尺寸 300/416
// top[0]->width();
///// yolo格式======================
// =====================================================
// top[1] 为 标签区 label 需要以不同格子大小区分()=======13*13格子======
// top_label ： batch->multi_label_[i]->mutable_cpu_data() 
//   N*150 ======
//  150 = 30*(1+4)  预设 30个边框空间，5个位一组边框参数，边框数量不足30， 后面填0============
// top[2]======================================26*26格子============


#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
///////////////////////////////////////////
#include "caffe/layers/annotated_box_data_layer.hpp"
///////////////////////////////////////
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {
//          reader_ (读数据到队列 data_reader ) ->
//  (prefetch 预读取 batch的数量    batch一次读取图像数量) 
//  (shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));// 数据库对象)
//  (db->Open(param_.data_param().source(), db::READ);// 打开数据库文件)
// BaseDataLayer  ->  BasePrefetchingDataLayer  -> AnnotatedBoxDataLayer  
// prefetch_free_(), prefetch_full_(), InternalThread 多线程读取数据
// transform_param_(数据转换，去均值处理)  -> BaseDataLayer
// 类构造函数=================================
template <typename Dtype>
AnnotatedBoxDataLayer<Dtype>::AnnotatedBoxDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),// 多线程读取+数据预处理
    reader_(param) // (读数据到队列 data_reader.cpp ) 
{
}

// 类析构函数=================================
template <typename Dtype>
AnnotatedBoxDataLayer<Dtype>::~AnnotatedBoxDataLayer() {
  this->StopInternalThread();
}

// AnnotatedBoxDataLayer 层初始化================
template <typename Dtype>
void AnnotatedBoxDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, // 这里底部没有数据过来
	const vector<Blob<Dtype>*>& top)// 输出数据和标签 
{
// 批次大小
  const int batch_size = this->layer_param_.data_param().batch_size();

// 获取 prototxtx 文件传入的 AnnotatedData 层的 annotated_data_param参数  
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
	  
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  
  // 类名字 ： 类id 映射prototxt文件=======================
  label_map_file_ = anno_data_param.label_map_file();


///////////////////////////////////////////////////////
  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  if (transform_param.has_resize_param()) 
  {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }
////////////////////////////////////////// 

// Read a data point, and use it to initialize the top blob.
  AnnotatedDatum& anno_datum = *(reader_.full().peek());
// 从队列里面读取 一个AnnotatedDatum数据节点

// 设置图像像素域============================================================= 
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
   // 为图像像素域 datum() 形状
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);// 图片数量N * 图片data
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
  {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  // N * 3 * 300 * 300 / N *3 * 416 *416  图像数据
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
	  
// 转换检测数据AnnotatedDatum
// 图像数据
// datum = anno_datum->mutable_datum()
//  datum->set_data();                      // 所有8位像素数据
//  datum->set_channels(cv_img.channels()); // 通道数量
//  datum->set_height(cv_img.rows);         // 行
//  datum->set_width(cv_img.cols);          // 列
//  datum->set_encoded(true);               // 编码?
// 标签
// anno_datum->mutable_annotation_group(g)->set_group_label(label)   标签
//                                        ->add_annotation()->mutable_bbox()  边框数据 
// 设置标签域 label===================================================================
  if (this->output_labels_) 
  {
/* 
    has_anno_type_ = anno_datum.has_type();
    vector<int> label_shape(4, 1);
	// 1*1*n*8， n=num_bboxes 为图像的实际标注框数量
    if (has_anno_type_) 
	{
      anno_type_ = anno_datum.type();
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);// 不同图片 不同的标注框数量
        label_shape[3] = 8;
      } 
	  else 
	  {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } 
	else 
	{
      label_shape[0] = batch_size;
    }
*/
////////// 
    //for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
	//{
    //  this->prefetch_[i].label_.clear(); // 预数据 清理
    //}
	
    vector<int> label_shape(1, batch_size);//存储了 1个 batch_size
    int label_size = (30 * 5); //(maxboxes=30)*(4+1)  最多30个边框 每个边框 类别id + box_[4] 
    label_shape.push_back(label_size);
    top[1]->Reshape(label_shape);// N * 150
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
	{
      this->prefetch_[i].label_.Reshape(label_shape);
      //shared_ptr<Blob<Dtype> > tmp_blob;
      //tmp_blob.reset(new Blob<Dtype>(label_shape));// N*150
      //this->prefetch_[i].multi_label_.push_back(tmp_blob); 
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//  读取数据库文件到队列，从队列中取数据，生成处理后的 批数据 Batch
// batch->data_                      为多个图像数据域
// batch->label_.mutable_cpu_data(); 为标签  每8个为1个边框标签
// top_label[idx++] = item_id;// batch id 0~3  / 0~8 / 0~15 图片id
// top_label[idx++] = anno_group.group_label();// 类别标签 id
// top_label[idx++] = anno.instance_id();      // 物体个数 实例id     
// top_label[idx++] = bbox.xmin();             // 左上角  
// top_label[idx++] = bbox.ymin();
// top_label[idx++] = bbox.xmax();             // 右下角
// top_label[idx++] = bbox.ymax();
// top_label[idx++] = bbox.difficult();        // ? 数据难度??
//////////////////////////////////////////////////////////////////////

// This function is called on prefetch thread
template<typename Dtype>
void AnnotatedBoxDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) 
{
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();// 批次大小图片数量
  
  /////////////////////===================================
   const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  
  /////////////////////=====================================
  AnnotatedDatum& anno_datum = *(reader_.full().peek());// 从队列读取数据================
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;      //  批次大小图片数量
  batch->data_.Reshape(top_shape);//  图像数据域=======================

  Dtype* top_data = batch->data_.mutable_cpu_data();
  
/////////////////////////////////////////////////////////////////
Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
// vector<Dtype*> top_label;
  
  if (this->output_labels_ && !has_anno_type_) 
  {
    // 标签数据
     top_label = batch->label_.mutable_cpu_data();
    //top_label.push_back(batch->multi_label_[i]->mutable_cpu_data());
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;// 每张图片的 边框数量
  
// 数据预处理============================================================= 
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a anno_datum
	// 从队列里 读取一个批次数据=======================================
    AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data from queue"));
    read_time += timer.MicroSeconds();
    timer.Start();
	
    // AnnotatedDatum distort_datum; // 图像增强处理 
	// 色相 Hue 、 饱和度 Saturation 、 明亮度 Lightness） 对比度 Contrast
	////// 这里未执行 expand 0像素扩展处理
	// AnnotatedDatum* expand_datum = NULL;// expand_datum 成后的数据
	/*
	// distort （变形亮度变化） 处理==================================
// 图像的HSL（ 色相 Hue 、 饱和度 Saturation 、 明亮度 Lightness） 对比度 Contrast
// random_order_prob  重新排布 图像通道 rgb 顺序
// expand 0像素扩展处理
    if (transform_param.has_distort_param()) 
	{
      distort_datum.CopyFrom(anno_datum);// 复制 anno_datum 到 distort_datum
	  // （变形亮度变化） 处理
      this->data_transformer_->DistortImage(anno_datum.datum(),
                                            distort_datum.mutable_datum());
      if (transform_param.has_expand_param()) 
	  {
        expand_datum = new AnnotatedDatum();
		// 这个主要是将DistortImage的图片用像素0进行扩展，标签bbox此时肯定会改变，
		// 就重新以黑边的左上角为原点计算[0,1]的bbox的左上角和右下角两个点坐标。
        this->data_transformer_->ExpandImage(distort_datum, expand_datum);
      } 
	  else 
	  {
        expand_datum = &distort_datum;
      }
    }
	
// 这个主要是将DistortImage的图片用像素0进行扩展，标签bbox此时肯定会改变，
// 就重新以黑边的左上角为原点计算[0,1]的bbox的左上角和右下角两个点坐标。
	else 
	{
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(anno_datum, expand_datum);
      } else {
        expand_datum = &anno_datum;
      }
    }
	
	*/
	
////////////////////////////////////////////////////////////////////////////
// 标签裁剪也很好理解首先要，
// 通过ProjectBBox将原坐标系标签投影到裁剪后图片的新坐标系的坐标，
// 然后再ClipBBox到通过ProjectBBox将原坐标系标签投影到裁剪后图片的新坐标系的坐标，
// 然后再ClipBBox到[0,1]之间。
    AnnotatedDatum* sampled_datum = NULL;
    if (batch_samplers_.size() > 0) 
	{
      // Generate sampled bboxes from anno_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(anno_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) 
	  {
        // Randomly pick a sampled bbox and crop the anno_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
		sampled_datum = new AnnotatedDatum();
        this->data_transformer_->CropImage(anno_datum, // 对象
		                                   sampled_bboxes[rand_idx],
                                           sampled_datum);// 对象地址
      } 
	  else 
	  {
        sampled_datum->CopyFrom(anno_datum);
      }
    } 
	else 
	{
      sampled_datum->CopyFrom(anno_datum);
    }
	
///*
///////////////////////////////////////////////////////////////////////////////
// Resize：最后将图片放缩到某个尺寸(300x300、416*416)，标签框也是线性放缩坐标而已。
    CHECK(sampled_datum != NULL);
    timer.Start();
    vector<int> shape =
        this->data_transformer_->InferBlobShape(sampled_datum->datum());

///////////////////
	if (transform_param.has_resize_param()) 
	{
      if (transform_param.resize_param().resize_mode() ==
          ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        this->transformed_data_.Reshape(shape);
        batch->data_.Reshape(shape);
        top_data = batch->data_.mutable_cpu_data();
      } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
              shape.begin() + 1));
      }
    } 
	else 
	{
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
            shape.begin() + 1));
    }
//*/


// 图片镜像、缩放、剪裁等===========================================
// Apply data transformations (mirror, scale, crop...)
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
	
    this->transformed_data_.set_cpu_data(top_data + offset);
	
//////////////////////////////////////////////////////////////////
///////////// 变换图像和标签 ////////////////////////
    vector<AnnotationGroup> transformed_anno_vec;
	
/////////////////////  data_transformer.cpp中添加 BoxLabel =========
	vector<BoxLabel> box_labels; 
	
	/////////////////////////////////////////////////
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
        CHECK_EQ(anno_type_, sampled_datum->type()) <<
            "Different AnnotationType.";
			
		/////////////////////////
			
			
        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();
		box_labels.clear();
		// 转换数据 和 标签=====================================
        this->data_transformer_->Transform(*sampled_datum,
                                           &(this->transformed_data_),
                                           &transformed_anno_vec);
        
		
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) 
		  {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } 
		else 
		{
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
      } 
	  else 
	  {
		  // 转换数据=====================
        this->data_transformer_->Transform(sampled_datum->datum(),
                                           &(this->transformed_data_));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
        top_label[item_id] = sampled_datum->datum().label();
      }
    } 
	else 
	{
		// 转换数据=====================
      this->data_transformer_->Transform(sampled_datum->datum(),
                                         &(this->transformed_data_));
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
  }

///// 内存清理方式还有点不同/////////////////////
/*
 // clear memory=====================================================
    if (has_sampled) 
	{
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) 
	{
      delete expand_datum;
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
}
*/

// 转换检测数据AnnotatedDatum
// 图像数据
// datum = anno_datum->mutable_datum()
//  datum->set_data();                      // 所有8位像素数据
//  datum->set_channels(cv_img.channels()); // 通道数量
//  datum->set_height(cv_img.rows);         // 行
//  datum->set_width(cv_img.cols);          // 列
//  datum->set_encoded(true);               // 编码?
// 标签
// anno_datum->mutable_annotation_group(g)->set_group_label(label)   标签
//                                        ->add_annotation()->mutable_bbox()  边框数据
 
  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_)
   {
    vector<int> label_shape(2);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) 
	{
		  
      label_shape[0] = batch_size;
	  label_shape[1] = 150;// 30*5 //
      // label_shape[3] = 8;
	  
      if (num_bboxes == 0) {  // 无边框 标签label 
        // Store all -1 in the label.
        //label_shape[2] = 1;   // 1*1*1*8========================
        //batch->label_.Reshape(label_shape);
         caffe_set<Dtype>(batch_size*150, 0, batch->label_.mutable_cpu_data());
      } 
	  
	  else 
	  { // 1*N*1*8===========================================
        // Reshape the label and store the annotation.
        //label_shape[2] = num_bboxes;// 多个边框数据
        batch->label_.Reshape(label_shape);
		
        top_label = batch->label_.mutable_cpu_data();
		
		caffe_set<Dtype>(batch_size*150, 0, top_label);
		
        int idx = 0;
		
        for (int item_id = 0; item_id < batch_size; ++item_id) 
		{ // 为 多个 图像的 标签 赋值
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) 
		  {
            const AnnotationGroup& anno_group = anno_vec[g];// 组
            for (int a = 0; a < anno_group.annotation_size(); ++a) 
			{
              const Annotation& anno = anno_group.annotation(a);// 类别
              const NormalizedBBox& bbox = anno.bbox();// 边框数据
              top_label[idx++] = anno_group.group_label();// 类比标签 id
              //top_label[idx++] = bbox.xmin();// 左上角 
              //top_label[idx++] = bbox.ymin();
              //top_label[idx++] = bbox.xmax();// 右下角
              //top_label[idx++] = bbox.ymax();
              //top_label[idx++] = bbox.difficult();
              top_label[idx++] = (Dtype)(bbox.xmin() + (bbox.xmax()-bbox.xmin())/2.);// 中点
              top_label[idx++] = (Dtype)(bbox.ymin() + (bbox.ymax()-bbox.ymin())/2.);;// 
              top_label[idx++] = (Dtype)(bbox.xmax()-bbox.xmin());// 尺寸
              top_label[idx++] = (Dtype)(bbox.ymax()-bbox.ymin());
            }
          }
        }
      }
    } 
	else 
	{
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
  
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(AnnotatedBoxDataLayer);// 层名字 AnnotatedBoxDataLayer
REGISTER_LAYER_CLASS(AnnotatedBoxData);  // 去掉 Layer --> AnnotatedBoxData

}  // namespace caffe
