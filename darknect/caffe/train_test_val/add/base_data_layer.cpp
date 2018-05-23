//////////////////////////////////////////////////////////
#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  
  
//////////////////////////////////////////
/////////////////////// add  ////////////////
  box_label_ = false;
//////////////////////////////////////////////
//////////////////////////////////////////////


  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_(param.data_param().prefetch()),
      prefetch_free_(), prefetch_full_(), prefetch_current_() {
  for (int i = 0; i < prefetch_.size(); ++i) {// vector的大小 prefetch_.size()
    prefetch_[i].reset(new Batch<Dtype>());
    prefetch_free_.push(prefetch_[i].get());
  }
}
///////////////////////// 这里函数不一样 需要查阅细节 主要是由于 
///////  prefetch_ 由数组 变成了 指针向量
/*
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
}
*/

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.


////// 重大变化   一个图像 多了许多边框标签////////////////
  for (int i = 0; i < prefetch_.size(); ++i) {// 数量变指针向量
// 使用  prefetch_.size()获取 向量大小
    prefetch_[i]->data_.mutable_cpu_data();// .访问变成 -> 访问
    if (this->output_labels_) {
      if (this->box_label_) {// 一张图片对应多个边框标签
        for (int j = 0; j < top.size() - 1; ++j) {
          prefetch_[i]->multi_label_[j]->mutable_cpu_data();
        }
      } else {
        prefetch_[i]->label_.mutable_cpu_data();
      }
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < prefetch_.size(); ++i) {
// 使用  prefetch_.size()获取 向量大小
      prefetch_[i]->data_.mutable_gpu_data();
      if (this->output_labels_) {
        if (this->box_label_) {// 一张图片对应多个边框标签
          for (int j = 0; j < top.size() - 1; ++j) {
            prefetch_[i]->multi_label_[j]->mutable_gpu_data();
          }
        } else {
          prefetch_[i]->label_.mutable_gpu_data();
        }
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        if (this->output_labels_) {
          batch->label_.data().get()->async_gpu_push(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}
//////////  重大变化   一个图像 多了许多边框标签////////////////
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
// Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
// 这里 prefetch_current_ 就相当于 batch 是一个指针
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  // Copy the data
///////// 这里不太一样  //////////////
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
// caffe_copy(prefetch_current_->data_.count(), 
//            prefetch_current_->data_.cpu_data(),
//            top[0]->mutable_cpu_data());

  if (this->output_labels_) {
/////////////////////////////  这里做了改动
    if (this->box_label_) {
      for (int i = 0; i < top.size() - 1; ++i) {
         top[i+1]->ReshapeLike(*(prefetch_current_->multi_label_[i]));
         top[i+1]->set_cpu_data(prefetch_current_->multi_label_[i]->mutable_cpu_data());
 //       top[i+1]->ReshapeLike(*(prefetch_current_->multi_label_[i]));
 //       caffe_copy(prefetch_current_->multi_label_[i]->count(), 
 //                 prefetch_current_->multi_label_[i]->cpu_data(),
 //                 top[i+1]->mutable_cpu_data());
      }
    } else {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
    }
  }
////////   add  是否与一开始重复  ////////////////////////
  prefetch_free_.push(prefetch_current_);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
