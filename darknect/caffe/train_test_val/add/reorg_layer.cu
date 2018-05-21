#include <algorithm>
#include <vector>

#include "caffe/layers/reorg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    template<typename Dtype>
    __global__ void Reorg(const int nthreads, const Dtype *in_data,
                          const bool reverse, const int stride, const int width,
                          const int height, const int channels, const int batch_num, Dtype *out_data) {
        int channels_out = channels / (stride * stride);
        int width_out = width * stride;
        int height_out = height * stride;
        int c_h_w = channels * width * height;
        int w_h = width * height;
        CUDA_KERNEL_LOOP(index, nthreads) {
            int index_batch_num = index / (c_h_w);
            int index_channels = index % (c_h_w) / (w_h);
            int index_height = index % (c_h_w) % (w_h) / width;
            int index_width = index % (c_h_w) % (w_h) % width;
            int c2 = index_channels % channels_out;
            int offset = index_channels / channels_out;
            int w2 = index_width * stride + offset % stride;
            int h2 = index_height * stride + offset / stride;
//            int out_index = ((((batch_num * channels_out + index_batch_num) + c2) * height_out + h2) * width_out) + w2;
            int out_index = ((index_batch_num * channels_out + c2) * height_out + h2) * width_out + w2;
            if (reverse) out_data[out_index] = in_data[index];
            else out_data[index] = in_data[out_index];
        }
    }

    template<typename Dtype>
    void ReorgLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
        const Dtype *bottom_data = bottom[0]->gpu_data();
        Dtype *top_data = top[0]->mutable_gpu_data();
        const int nthreads = bottom[0]->count();
        Reorg<Dtype>
                <<< CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >>> (
                nthreads, bottom_data, reverse_, stride_, width_, height_,
                        channels_, batch_num_, top_data);

    }
    template <typename Dtype>
    void ReorgLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
        if(!propagate_down[0]){
            return;
        }
        const Dtype *top_diff = top[0]->gpu_diff();
        Dtype *bottom_diff = top[0]->mutable_gpu_diff();
        const int nthreads = bottom[0]->count();
        Reorg<Dtype>
            <<< CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >>>(
                nthreads, top_diff, !reverse_, stride_, width_, height_,
                channels_, batch_num_, bottom_diff);
    }
    INSTANTIATE_LAYER_GPU_FUNCS(ReorgLayer);
} // namespace caffe
