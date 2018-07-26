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

  

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {
//          reader_ (读数据到队列 data_reader ) ->
//  (prefetch 预读取 batch的数量    batch一次读取图像数量) 
//  (shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));// 数据库对象)
//  (db->Open(param_.data_param().source(), db::READ);// 打开数据库文件)
// BaseDataLayer  ->  BasePrefetchingDataLayer  -> AnnotatedDataLayer 
// prefetch_free_(), prefetch_full_(), InternalThread 多线程读取数据
// transform_param_(数据转换，去均值处理)  -> BaseDataLayer
// 类构造函数=================================
template <typename Dtype>
AnnotatedDataLayer<Dtype>::AnnotatedDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),// 多线程读取+数据预处理
    reader_(param) // (读数据到队列 data_reader.cpp ) 
{
}
// 类析构函数=================================
template <typename Dtype>
AnnotatedDataLayer<Dtype>::~AnnotatedDataLayer() {
  this->StopInternalThread();
}

// AnnotatedDataLayer 层初始化================
template <typename Dtype>
void AnnotatedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, // 这里地步没有数据过来
	const vector<Blob<Dtype>*>& top) // 输出数据和标签
{
  const int batch_size = this->layer_param_.data_param().batch_size();
  
// 获取 prototxtx 文件传入的 AnnotatedData 层的 annotated_data_param参数
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param(); 
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  
// 类名字 ： 类id 映射prototxt文件=======================
  label_map_file_ = anno_data_param.label_map_file();
  
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

  // Read a data point, and use it to initialize the top blob.
  // 从堆里里面读取 一个AnnotatedDatum数据节点 
  AnnotatedDatum& anno_datum = *(reader_.full().peek());

  // 设置图像像素域============================================
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
      // 为图像像素域 datum() 形状
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  
  top_shape[0] = batch_size;// 图片数量N * 图片data
  top[0]->Reshape(top_shape);
  
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {// 3 队列大小??
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

   
  // 设置标签域 label==================================================
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);// 
    if (has_anno_type_) 
	{
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) 
	  {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
	  
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) 
	  {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
		//  类别分组 
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) 
		{
          num_bboxes += anno_datum.annotation_group(g).annotation_size();// 总边框数量
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);// 预测边框数量
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
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
	{
      this->prefetch_[i].label_.Reshape(label_shape);
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
void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  AnnotatedDatum& anno_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);//  图像数据域=======================

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;

// 数据预处理============================================================= 
  for (int item_id = 0; item_id < batch_size; ++item_id) 
  {
    timer.Start();
    // get a anno_datum
    AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum = NULL;// expand_datum 成后的数据
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
	
// 标签裁剪也很好理解首先要，
// 通过ProjectBBox将原坐标系标签投影到裁剪后图片的新坐标系的坐标，
// 然后再ClipBBox到通过ProjectBBox将原坐标系标签投影到裁剪后图片的新坐标系的坐标，
// 然后再ClipBBox到[0,1]之间。
    AnnotatedDatum* sampled_datum = NULL;
    bool has_sampled = false;
    if (batch_samplers_.size() > 0) 
	{
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new AnnotatedDatum();
        this->data_transformer_->CropImage(*expand_datum,
                                           sampled_bboxes[rand_idx],
                                           sampled_datum);
        has_sampled = true;
      } else {
        sampled_datum = expand_datum;
      }
    } 
	else 
	{
      sampled_datum = expand_datum;
    }
	
    CHECK(sampled_datum != NULL);
    timer.Start();
    vector<int> shape =
        this->data_transformer_->InferBlobShape(sampled_datum->datum());
		
///////////////////////////////////////////////////////////////////////////////
// Resize：最后将图片放缩到某个尺寸(300x300、416*416)，标签框也是线性放缩坐标而已。
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
	
// 图片镜像、缩放、剪裁等===========================================
// Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
	
    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) 
	{
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) 
		{
          sampled_datum->set_type(anno_type_);
        } 
		else 
		{
          CHECK_EQ(anno_type_, sampled_datum->type()) <<
              "Different AnnotationType.";
        }
        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();
        this->data_transformer_->Transform(*sampled_datum,
                                           &(this->transformed_data_),
                                           &transformed_anno_vec);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
      } else {
        this->data_transformer_->Transform(sampled_datum->datum(),
                                           &(this->transformed_data_));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
        top_label[item_id] = sampled_datum->datum().label();
      }
    } 
	else 
	{
      this->data_transformer_->Transform(sampled_datum->datum(),
                                         &(this->transformed_data_));
    }
	
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
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = 8;
      // 无边框 标签label 
      if (num_bboxes == 0) 
	  {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_.Reshape(label_shape);
        caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
      } 
	  else 
	  {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;// 多个边框数据
        batch->label_.Reshape(label_shape);
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) 
		  {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a)
				{
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();// 边框数据
              top_label[idx++] = item_id;// batch id 0~3  / 0~8 / 0~15
              top_label[idx++] = anno_group.group_label();// 类比标签 id
              top_label[idx++] = anno.instance_id();
              top_label[idx++] = bbox.xmin();// 左上角  
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();// 右下角
              top_label[idx++] = bbox.ymax();
              top_label[idx++] = bbox.difficult();
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

INSTANTIATE_CLASS(AnnotatedDataLayer);
REGISTER_LAYER_CLASS(AnnotatedData);

}  // namespace caffe
