#ifndef CAFFE_REORG_LAYER_HPP_
#define CAFFE_REORG_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief Reshapes the input Blob into an arbitrary-sized output Blob.
 *
 * Note: similarly to FlattenLayer, this layer does not change the input values
 * (see FlattenLayer, Blob::ShareData and Blob::ShareDiff).
 */
    template<typename Dtype>
    class ReorgLayer : public Layer<Dtype> {
    public:
        explicit ReorgLayer(const LayerParameter &param)
                : Layer<Dtype>(param) {}

        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top);

        virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

        virtual inline const char *type() const { return "Reorg"; }

        virtual inline int ExactNumBottomBlobs() const { return 1; }

        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:


        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        int stride_;
        bool reverse_;
        int batch_num_;
        int channels_;
        int reorged_channels_;
        int height_, width_;
        int reorged_height_, reorged_width_;
    };


}  // namespace caffe

#endif  // CAFFE_REORG_LAYER_HPP_
