#include <algorithm>
#include <vector>

#include "caffe/layers/region_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    template<typename Dtype>
    __global__ void flatten_gpu(const int nthreads, const Dtype *in_data,
                                const int area, const int channel, const int batch,
                                const bool forward, Dtype *out_data) {
        int c_a = channel * area;
        CUDA_KERNEL_LOOP(index, nthreads) {
            int b = index / c_a;
            int c = index % c_a / area;
            int i = index % c_a % area;
            int in = b * c_a + c * area + i;
            int out = b * c_a + i * channel + c;
            if (forward) out_data[out] = in_data[in];
            else out_data[in] = in_data[out];
        }
    }

    template<typename Dtype>
    __global__ void logistics_gpu(const int nthreads, const int size,
                                  const int h_w_b, const int h_w_c, Dtype *data) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            int b = index / h_w_c;
            int i = index % h_w_b;
            int in_index = size * i + b * h_w_c;
            data[in_index + 4] = 1. / (1 + exp(-data[in_index + 4]));
        }
    }

    template<typename Dtype>
    __global__ void softmax_gpu(const int nthreads, const int height, const int width,
                                const int channel, const int classes, const int boxes_of_each_grid, Dtype *out_data) {

        CUDA_KERNEL_LOOP(index, nthreads) {
            int b = index / (height * width * boxes_of_each_grid);
            int i = index % (height * width * boxes_of_each_grid);
            int index_2 = 85 * i + b * height * width * channel;
            Dtype sum = 0;
            Dtype largest = -1000;
            for (int j = 0; j < classes; j++) {
                if (out_data[j + index_2 + 5] > largest) largest = out_data[j + index_2 + 5];
            }
            for (int j = 0; j < classes; j++) {
                Dtype e = exp(out_data[j + index_2 + 5] - largest);
                sum += e;
                out_data[j + index_2 + 5] = e;
            }
            for (int j = 0; j < classes; j++) {
                out_data[j + index_2 + 5] /= sum;
            }
        }
    }

    template<typename Dtype>
    void RegionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                         const vector<Blob<Dtype> *> &top) {
        const Dtype *bottom_data = bottom[0]->gpu_data();
        Dtype *top_data = top[0]->mutable_gpu_data();
        int nthreads = bottom[0]->count();
        int size = coords_ + classes_ + 1;//每个box需要4个坐标值，一个scale，80个种类的概率
        nthreads = bottom[0]->count();
        flatten_gpu<Dtype>
                << < CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >
                                                 (nthreads, bottom_data, width_ * height_,
                                                         boxes_of_each_grid_ * size, batch_num_, true, top_data);
        nthreads = batch_num_ * height_ * width_ * boxes_of_each_grid_;
        logistics_gpu<Dtype>
                << < CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >
                                                 (nthreads, size, height_ * width_ *
                                                                  boxes_of_each_grid_, bottom[0]->count(), top_data);
        if (softmax_) {
            softmax_gpu<Dtype>
                    << < CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >
                                                     (nthreads, height_, width_, channels_,
                                                             classes_, boxes_of_each_grid_, top_data);
        }
    }

    template<typename Dtype>
    void RegionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                          const vector<Blob<Dtype> *> &bottom) {
        if (!propagate_down[0]) {
            return;
        }
        const Dtype *top_diff = top[0]->gpu_diff();
        Dtype *bottom_diff = top[0]->mutable_gpu_diff();
        const int nthreads = bottom[0]->count();
    }

    INSTANTIATE_LAYER_GPU_FUNCS(RegionLayer);
} // namespace caffe
