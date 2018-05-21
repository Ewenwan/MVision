#include "caffe/layers/region_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template<typename Dtype>
    void RegionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
        CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
                    "allow in-place computation.";
        RegionParameter region_param = this->layer_param_.region_param();
        classes_ = region_param.classes();
        coords_ = region_param.coords();
        boxes_of_each_grid_ = region_param.boxes_of_each_grid();

        channels_ = bottom[0]->channels();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();
        batch_num_ = bottom[0]->num();
    }

    template<typename Dtype>
    void RegionLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
        top[0]->Reshape(batch_num_, channels_,
                        height_, width_);
    }

    template<typename Dtype>
    void RegionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                         const vector<Blob<Dtype> *> &top) {
        const Dtype *bottom_data = bottom[0]->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        caffe_copy(batch_num_ * channels_ * height_ * width_, bottom_data, top_data);
        int size = coords_ + classes_ + 1;//每个box需要4个坐标值，一个scale，80个种类的概率
        flatten(top_data, width_ * height_, boxes_of_each_grid_ * size, batch_num_, true);

        for (int b = 0; b < batch_num_; ++b) {
            for (int i = 0; i < height_ * width_ * boxes_of_each_grid_; ++i) {
                int index = size * i + b * height_ * width_ * channels_;
                top_data[index + 4] = logistic_activate(top_data[index + 4]);
            }
        }

        if (softmax_) {
            for (int b = 0; b < batch_num_; ++b) {
                for (int i = 0; i < height_ * width_ * boxes_of_each_grid_; ++i) {
                    int index = size * i + b * height_ * width_ * channels_;
                    softmax(top_data + index + 5, classes_, (Dtype) 1, top_data + index + 5);
                }
            }
        }
    }

    template<typename Dtype>
    void RegionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                          const vector<Blob<Dtype> *> &bottom) {
        if (!propagate_down[0]) {
            return;
        }
        const Dtype *top_diff = top[0]->cpu_diff();
        Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    }

    INSTANTIATE_CLASS(RegionLayer);

    REGISTER_LAYER_CLASS(Region);

}  // namespace caffe
