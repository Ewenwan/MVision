/////////////////////////////////// 也有大变换 ////////////////
#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

////// 重大变化   一个图像 多了许多边框标签////////////////
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

// Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
// 这里 prefetch_current_ 就相当于 batch 是一个指针
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  ///////// 这里不太一样  //////////////
  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
  // caffe_copy(prefetch_current_->data_.count(), 
  //            prefetch_current_->data_.cpu_data(),
  //            top[0]->mutable_cpu_data());
  if (this->output_labels_) {

/////////////////////////////  这里做了改动
     if (this->box_label_) {
      for (int i = 0; i < top.size() - 1; ++i) {
        // Reshape to loaded labels.
        top[i+1]->ReshapeLike(*(prefetch_current_->multi_label_[i]));
        // copy 
        top[i+1]->set_cpu_data(prefetch_current_->multi_label_[i]->mutable_gpu_data());
        //caffe_copy(prefetch_current_->multi_label_[i]->count(), 
        //           prefetch_current_->multi_label_[i]->gpu_data(),
        //           top[i+1]->mutable_gpu_data());

      }
    } else {
      // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
      // Copy the labels.
    top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
    // caffe_copy(prefetch_current_->label_.count(), 
    //            prefetch_current_->label_.gpu_data(),
    //            top[1]->mutable_gpu_data());
     }
  }
/////////////////// add 是否与一开始重复 //////////////////
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(prefetch_current_);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
