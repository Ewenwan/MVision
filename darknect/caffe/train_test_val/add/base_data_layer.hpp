//////////////////////////////////////////////
#ifndef CAFFE_BASE_DATA_LAYERS_HPP_
#define CAFFE_BASE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
//#include "caffe/layers/base_data_layer.hpp"
namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */

//////////////// 基本 数据层类型 添加一个 图像中 边框标签数据
//////////////// 图片分类任务 转成 目标检测 一个图像中需要检测出多个物体
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

////////////////////////////////////////////////////////////
  // Data layers should be shared by multiple solvers in parallel
  //virtual inline bool ShareInParallel() const { return true; }
///////////////////////////////////////////////////////

  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  bool output_labels_;
  
  //////////////////// add /////////////////
  bool box_label_;// 一个图像中需要检测出多个物体 边框 标签
  //////////////////////////////
  
};

template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;// 普通向量 .访问成员
  
///////////////// add /////////////////////////////////  
// vector<Blob<Dtype> > multi_label_;// 向量  .访问成员
  vector<shared_ptr<Blob<Dtype> > > multi_label_;// 指针向量 -> 访问成员
 ////////////////////////////////////////////////// 
 
};

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


//////  add /////////////////////////////////////
// Prefetches batches (asynchronously if to GPU memory)
// static const int PREFETCH_COUNT = 3;// 这里可能不需要了 prefetch_变成了 vector
////////////////////////////////////////////


 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch) = 0;
  
  //////////////////////////重要改变/////////////////////////////////////
 // Batch<Dtype> prefetch_[PREFETCH_COUNT];  // 数组 PREFETCH_COUNT 数组大小
 ////////////////////////////////// 
  
  vector<shared_ptr<Batch<Dtype> > > prefetch_;// 换成 指针vector 大小使用  prefetch_.size()获取
  //  .访问变成  ->访问
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;
  
  Batch<Dtype>* prefetch_current_;// 当前 Batch<Dtype> 数据指针

  Blob<Dtype> transformed_data_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_DATA_LAYERS_HPP_
