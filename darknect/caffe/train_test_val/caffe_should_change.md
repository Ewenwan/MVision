# caffe 需要做的修改
# 新添加的文件
# 1. 数据转换 时

    caffe/util/io.hpp 
    caffe/src/caffe/util/io.cpp 做一定修改  边框数据

    /tools v添加
    convert_box_data.cpp

# 2.训练时 
    a. 数据层
    caffe/include/caffe/layer 下添加 
      box_data_layer.hpp
    caffe/src/caffe/layers/  下添加 
      box_data_layer.cpp

      需要修改


    b. 检测loss层
      detection_loss_layer.hpp
      detection_loss_layer.cpp
      
    c. 评估检测结果层
      eval_detection_layer.hpp
      eval_detection_layer.cpp
      
    d. 数据读取层
    caffe/include/caffe 下添加
        data_reader.hpp
    caffe/src/caffe/ 下添加 
        data_reader.cpp

# 3. 添加一些工具
    caffe/tools 下添加
       a. 数据转换
        convert_box_data.cpp
       b. 设备 队列
        device_query.cpp
       c. 模型微调 (finetune)
        finetune_net.cpp
       d. 网络速度检测
        net_speed_benchmark.cpp
       e. 测试 检测结果
        test_detection.cpp
       f. 测试 网络 
        test_net.cpp
       g. 检测网络
        train_net.cpp

# 4. 添加新层的 一些注意事项

[参考](https://blog.csdn.net/shuzfan/article/details/51322976)

[参考2 ](https://blog.csdn.net/wfei101/article/details/76735760)

    每一种层都对应一个同名cpp和hpp文件
    分别在:
    caffe/include/caffe/layer 下  .cpp
    caffe/src/caffe/layers/  下   .hpp

## 4.1 例如 ：头文件编写 pooling_layer.hpp

    // pooling_layer.hpp
    // 特别注意：命名的时候应严格一致和注意大小写，这一点是导致很多人加层失败的主要原因。
    #ifndef CAFFE_POOLING_LAYER_HPP_
    #define CAFFE_POOLING_LAYER_HPP_

    #include <vector>

    #include "caffe/blob.hpp"
    #include "caffe/layer.hpp"
    #include "caffe/proto/caffe.pb.h"

    namespace caffe {

    /**
     * @brief Pools the input image by taking the max, average, etc. within regions.
     *
     * TODO(dox): thorough documentation for Forward, Backward, and proto params.
     */
    template <typename Dtype>
    // 以后我们层的type: "Pooling" =========================
    class PoolingLayer : public Layer<Dtype> {
     public:
        // 类 构造函数
      explicit PoolingLayer(const LayerParameter& param)
          : Layer<Dtype>(param) {}
        // 类初始化函数
      virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);
        //  根据 bottom的shape 和池化参数 修改top的shape 
      virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);

      // 以后我们层的type: "Pooling"  ====================================
      virtual inline const char* type() const { return "Pooling"; }

      // 我们只需要一个bottom和一个top
      virtual inline int ExactNumBottomBlobs() const { return 1; }

      virtual inline int MinTopBlobs() const { return 1; }
      // MAX POOL layers can output an extra top blob for the mask;
      // others can only output the pooled inputs.
      virtual inline int MaxTopBlobs() const {
        return (this->layer_param_.pooling_param().pool() ==
                PoolingParameter_PoolMethod_MAX) ? 2 : 1;
      }

     protected:
     // 前向 & 反向传播 的 CPU/GPU 版本
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);
      virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
      virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

      int kernel_h_, kernel_w_;// 池化核 尺寸
      int stride_h_, stride_w_;// 滑动步长
      int pad_h_, pad_w_;      // 填充
      int channels_;           // 输入特征通道数量
      int height_, width_;     // 输入特征图尺寸
      int pooled_height_, pooled_width_;
      bool global_pooling_;
      // 定义一个存储 Dtype型的 Blob
      Blob<Dtype> rand_idx_;
      Blob<int> max_idx_;
    };

    }  // namespace caffe

    #endif  // CAFFE_POOLING_LAYER_HPP_

## 4.2 例如 ：源文件编写 pooling_layer.cpp
    // pooling_layer.cpp
    #include <algorithm>
    #include <cfloat>
    #include <vector>

    #include "caffe/layers/pooling_layer.hpp"
    #include "caffe/util/math_functions.hpp"

    namespace caffe {

    using std::min;
    using std::max;

    template <typename Dtype>
    // 层初始化函数 
    void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top) {
      PoolingParameter pool_param = this->layer_param_.pooling_param();
      if (pool_param.global_pooling()) {// 全局池化

        CHECK(!(pool_param.has_kernel_size() ||
          pool_param.has_kernel_h() || pool_param.has_kernel_w()))
          << "With Global_pooling: true Filter size cannot specified";
      } 
      else {// 普通 池化
        CHECK(!pool_param.has_kernel_size() !=
          !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
          << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
        CHECK(pool_param.has_kernel_size() ||
          (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
          << "For non-square filters both kernel_h and kernel_w are required.";
      }
      // 填充
      CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
          && pool_param.has_pad_w())
          || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
          << "pad is pad OR pad_h and pad_w are required.";
      // 步长
      CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
          && pool_param.has_stride_w())
          || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
          << "Stride is stride OR stride_h and stride_w are required.";
      global_pooling_ = pool_param.global_pooling();
      // 全局池化
      if (global_pooling_) {
        kernel_h_ = bottom[0]->height();
        kernel_w_ = bottom[0]->width();
      } 
      // 普通 池化
      else {
        if (pool_param.has_kernel_size()) {
          kernel_h_ = kernel_w_ = pool_param.kernel_size();
        } else {
          kernel_h_ = pool_param.kernel_h();
          kernel_w_ = pool_param.kernel_w();
        }
      }
      CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
      CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
      if (!pool_param.has_pad_h()) {
        pad_h_ = pad_w_ = pool_param.pad();
      } else {
        pad_h_ = pool_param.pad_h();
        pad_w_ = pool_param.pad_w();
      }
      if (!pool_param.has_stride_h()) {
        stride_h_ = stride_w_ = pool_param.stride();
      } else {
        stride_h_ = pool_param.stride_h();
        stride_w_ = pool_param.stride_w();
      }
      if (global_pooling_) {
        CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
          << "With Global_pooling: true; only pad = 0 and stride = 1";
      }
      if (pad_h_ != 0 || pad_w_ != 0) {
        // 均值池化 /  最大值池化 
        CHECK(this->layer_param_.pooling_param().pool()
            == PoolingParameter_PoolMethod_AVE
            || this->layer_param_.pooling_param().pool()
            == PoolingParameter_PoolMethod_MAX)
            << "Padding implemented only for average and max pooling.";
        CHECK_LT(pad_h_, kernel_h_);
        CHECK_LT(pad_w_, kernel_w_);
      }
    }

    //  根据 bottom的shape 和池化参数 修改top的shape 
    template <typename Dtype>
    void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top) {
      CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
          << "corresponding to (num, channels, height, width)";
      channels_ = bottom[0]->channels();
      height_ = bottom[0]->height();
      width_ = bottom[0]->width();
      if (global_pooling_) {
        kernel_h_ = bottom[0]->height();
        kernel_w_ = bottom[0]->width();
      }
      pooled_height_ = static_cast<int>(ceil(static_cast<float>(
          height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
      pooled_width_ = static_cast<int>(ceil(static_cast<float>(
          width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
      if (pad_h_ || pad_w_) {
        // If we have padding, ensure that the last pooling starts strictly
        // inside the image (instead of at the padding); otherwise clip the last.
        if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
          --pooled_height_;
        }
        if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
          --pooled_width_;
        }
        CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
        CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
      }
      top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
          pooled_width_);
      if (top.size() > 1) {
        top[1]->ReshapeLike(*top[0]);
      }
      // If max pooling, we will initialize the vector index part.
      if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_MAX && top.size() == 1) {
        max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
            pooled_width_);
      }
      // If stochastic pooling, we will initialize the random index part.
      if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_STOCHASTIC) {
        rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
          pooled_width_);
      }
    }

    // 前向传播 CPU版本=========================
    // TODO(Yangqing): Is there a faster way to do pooling in the channel-first
    // case?
    template <typename Dtype>
    void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top) {
      const Dtype* bottom_data = bottom[0]->cpu_data();
      Dtype* top_data = top[0]->mutable_cpu_data();
      const int top_count = top[0]->count();
      // We'll output the mask to top[1] if it's of size >1.
      const bool use_top_mask = top.size() > 1;
      int* mask = NULL;  // suppress warnings about uninitalized variables
      Dtype* top_mask = NULL;
      // Different pooling methods. We explicitly do the switch outside the for
      // loop to save time, although this results in more code.
      switch (this->layer_param_.pooling_param().pool()) {
      case PoolingParameter_PoolMethod_MAX:
        // Initialize
        if (use_top_mask) {
          top_mask = top[1]->mutable_cpu_data();
          caffe_set(top_count, Dtype(-1), top_mask);
        } else {
          mask = max_idx_.mutable_cpu_data();
          caffe_set(top_count, -1, mask);
        }
        caffe_set(top_count, Dtype(-FLT_MAX), top_data);
        // The main loop
        for (int n = 0; n < bottom[0]->num(); ++n) {
          for (int c = 0; c < channels_; ++c) {
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                int hstart = ph * stride_h_ - pad_h_;
                int wstart = pw * stride_w_ - pad_w_;
                int hend = min(hstart + kernel_h_, height_);
                int wend = min(wstart + kernel_w_, width_);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                const int pool_index = ph * pooled_width_ + pw;
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    const int index = h * width_ + w;
                    if (bottom_data[index] > top_data[pool_index]) {
                      top_data[pool_index] = bottom_data[index];
                      if (use_top_mask) {
                        top_mask[pool_index] = static_cast<Dtype>(index);
                      } else {
                        mask[pool_index] = index;
                      }
                    }
                  }
                }
              }
            }
            // compute offset
            bottom_data += bottom[0]->offset(0, 1);
            top_data += top[0]->offset(0, 1);
            if (use_top_mask) {
              top_mask += top[0]->offset(0, 1);
            } else {
              mask += top[0]->offset(0, 1);
            }
          }
        }
        break;
      case PoolingParameter_PoolMethod_AVE:
        for (int i = 0; i < top_count; ++i) {
          top_data[i] = 0;
        }
        // The main loop
        for (int n = 0; n < bottom[0]->num(); ++n) {
          for (int c = 0; c < channels_; ++c) {
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                int hstart = ph * stride_h_ - pad_h_;
                int wstart = pw * stride_w_ - pad_w_;
                int hend = min(hstart + kernel_h_, height_ + pad_h_);
                int wend = min(wstart + kernel_w_, width_ + pad_w_);
                int pool_size = (hend - hstart) * (wend - wstart);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                hend = min(hend, height_);
                wend = min(wend, width_);
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    top_data[ph * pooled_width_ + pw] +=
                        bottom_data[h * width_ + w];
                  }
                }
                top_data[ph * pooled_width_ + pw] /= pool_size;
              }
            }
            // compute offset
            bottom_data += bottom[0]->offset(0, 1);
            top_data += top[0]->offset(0, 1);
          }
        }
        break;
      case PoolingParameter_PoolMethod_STOCHASTIC:
        NOT_IMPLEMENTED;
        break;
      default:
        LOG(FATAL) << "Unknown pooling method.";
      }
    }


    // 反向传播 CPU版本 =====================================
    template <typename Dtype>
    void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      if (!propagate_down[0]) {
        return;
      }
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      // Different pooling methods. We explicitly do the switch outside the for
      // loop to save time, although this results in more codes.
      caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
      // We'll output the mask to top[1] if it's of size >1.
      const bool use_top_mask = top.size() > 1;
      const int* mask = NULL;  // suppress warnings about uninitialized variables
      const Dtype* top_mask = NULL;
      switch (this->layer_param_.pooling_param().pool()) {
      case PoolingParameter_PoolMethod_MAX:
        // The main loop
        if (use_top_mask) {
          top_mask = top[1]->cpu_data();
        } else {
          mask = max_idx_.cpu_data();
        }
        for (int n = 0; n < top[0]->num(); ++n) {
          for (int c = 0; c < channels_; ++c) {
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                const int index = ph * pooled_width_ + pw;
                const int bottom_index =
                    use_top_mask ? top_mask[index] : mask[index];
                bottom_diff[bottom_index] += top_diff[index];
              }
            }
            bottom_diff += bottom[0]->offset(0, 1);
            top_diff += top[0]->offset(0, 1);
            if (use_top_mask) {
              top_mask += top[0]->offset(0, 1);
            } else {
              mask += top[0]->offset(0, 1);
            }
          }
        }
        break;
      case PoolingParameter_PoolMethod_AVE:
        // The main loop
        for (int n = 0; n < top[0]->num(); ++n) {
          for (int c = 0; c < channels_; ++c) {
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                int hstart = ph * stride_h_ - pad_h_;
                int wstart = pw * stride_w_ - pad_w_;
                int hend = min(hstart + kernel_h_, height_ + pad_h_);
                int wend = min(wstart + kernel_w_, width_ + pad_w_);
                int pool_size = (hend - hstart) * (wend - wstart);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                hend = min(hend, height_);
                wend = min(wend, width_);
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    bottom_diff[h * width_ + w] +=
                      top_diff[ph * pooled_width_ + pw] / pool_size;
                  }
                }
              }
            }
            // offset
            bottom_diff += bottom[0]->offset(0, 1);
            top_diff += top[0]->offset(0, 1);
          }
        }
        break;
      case PoolingParameter_PoolMethod_STOCHASTIC:
        NOT_IMPLEMENTED;
        break;
      default:
        LOG(FATAL) << "Unknown pooling method.";
      }
    }


    #ifdef CPU_ONLY
    STUB_GPU(PoolingLayer);
    #endif

    INSTANTIATE_CLASS(PoolingLayer);

    }  // namespace caffe

## 4.3 修改src\caffe\proto\caffe.proto文件
    这里我们要为我们新写的层添加层参数 和  层消息函数。
### 4.3.1 如果层有参数 需要在 message LayerParameter {} 中添加 新层 信息
    首先应该在message LayerParameter {}中添加新参数信息

    添加信息时，首先要制定一个唯一ID，这个ID的可选值可以由这句话看出：
    // LayerParameter next available layer-specific ID: 143 (last added: BatchCLuster)
    message LayerParameter {
    ... 

    由上图可以看出，可选的ID为143。 
    于是我们就可以添加这样一行：  
      optional DiffCutoffParameter diffcutoff_param = 143;

    yolo v1  检测层  和 评估层 
    ////////////  add ////////////
    // Yolo detection loss layer
    optional DetectionLossParameter detection_loss_param = 200;
    // Yolo detection evaluation layer
    optional EvalDetectionParameter eval_detection_param = 201;
    ////////////////// Add 


    yolov2   passthrough层 ReorgParameter  和  最后区域 Region层 RegionParameter

    ///////////////////////////////////// YOLO V2 ADD  /////////////
    optional ReorgParameter reorg_param = 148;
    optional RegionParameter region_param = 149;
    //////////////////////////////////////////////////////

### 4.3.2 在任意位置添加 层 消息函数
    message DiffCutoffParameter {
      optional float diff_scale = 1 [default = 1]; //默认梯度不缩放
    }

    //////////////////////////////////// yolov1 add  ///////////////
    message DetectionLossParameter {
      // Yolo detection loss layer
      optional uint32 side = 1 [default = 7];
      optional uint32 num_class = 2 [default = 20];
      optional uint32 num_object = 3 [default = 2];
      optional float object_scale = 4 [default = 1.0];
      optional float noobject_scale = 5 [default = 0.5];
      optional float class_scale = 6 [default = 1.0];
      optional float coord_scale = 7 [default = 5.0];
      optional bool sqrt = 8 [default = true];
      optional bool constriant = 9 [default = false];
    }
    ///////////////////////////////////////////////////
    message EvalDetectionParameter {
      enum ScoreType {
        OBJ = 0;
        PROB = 1;
        MULTIPLY = 2;
      }
      // Yolo detection evaluation layer
      optional uint32 side = 1 [default = 7];
      optional uint32 num_class = 2 [default = 20];
      optional uint32 num_object = 3 [default = 2];
      optional float threshold = 4 [default = 0.5];
      optional bool sqrt = 5 [default = true];
      optional bool constriant = 6 [default = true];
      optional ScoreType score_type = 7 [default = MULTIPLY];
      optional float nms = 8 [default = -1];
    }
    ////////////////////////////////// add ////////////


    ///////////////////////// YOLO V2  add ////////////////////////////////////
    // 区域参数
    message RegionParameter {
      optional uint32 classes = 1 [default = 20]; //分类的种类
      optional uint32 coords = 2 [default = 4]; //box的坐标数
      optional uint32 boxes_of_each_grid = 3 [default = 5]; //每个grid预测的boxes数
      optional bool softmax = 4 [default = false];
    }
    //  passtrough层
     message ReorgParameter {
       optional uint32 stride = 1;// 步长
       optional bool reverse = 2 [default = false];
     }
    ///////////////////////////////////////////////////////////////////////




### 4.3.3 在message V1LayerParameter {}中添加以下内容

    在enum LayerType {}中添加唯一ID，只要在这里不重复即可。

       DIFF_CUTOFF=45;

    外面接着添加，同样ID也是只要不重复即可

      optional DiffCutoffParameter diffcutoff_param = 46;

### 4.3.4  在message V0LayerParameter {}添加参数定义

      optional float diff_scale = 47 [default = 1]; 
## 4.4 最后重新编译caffe即可
    可添加层参数 ：
        layer {
          name: "diff_1"
          type: "DiffCutoff"
          bottom: "conv1"
          top: "diff_1"
          diffcutoff_param {
            diff_scale: 0.0001
          }
        }
## 4.5 忠告与建议
    （1）一定要注意大小写、一定要注意大小写、一定要注意大小写

    （2）不会写、不确定，就去找caffe现有的层来参考模仿

    （3）caffe数据操作的函数定义在src/caffe/util/math_functions.cpp, 

    大家也可以参考这位同学的博客  
(caffe数据操作的函数)[https://blog.csdn.net/seven_first/article/details/47378697]
    

## 4.6 修改的 文件有
    caffe\src\caffe\proto\caffe.proto

    caffe\include\caffe\layers\base_data_layer.hpp  
        class BaseDataLayer : public Layer<Dtype> {} // 添加 bool box_label_; 边框标签

        base_data_layer.c 
        BaseDataLayer<Dtype>::LayerSetUp{}  
        //////////////////////////////////////////
        /////////////////////// add  ////////////////
        box_label_ = false;
        //////////////////////////////////////////////
        //////////////////////////////////////////////


    caffe\include\caffe\layers\data_layer.hpp   // 可能不需要改 
       //////// add ///////////////////////////////
        #include "caffe/data_reader.hpp"
       //////// add ///////////////////////////////

    dummy_data_layer.hpp

        // /////  add //////////////////////////////
        // Data layers should be shared by multiple solvers in parallel
        // virtual inline bool ShareInParallel() const { return true; }
        /////////////////////////////////////////

    hdf5_data_layer.hpp      // 可能不需要改 
    hdf5_output_layer.hpp   // 可能不需要改 



    input_layer.hpp
        ///////  add  ///////////////////////////
          // Data layers should be shared by multiple solvers in parallel
        //virtual inline bool ShareInParallel() const { return true; }
        ///////  add ////////////////////////////////////


    caffe\include\caffe\data_transformer.hpp   需要修改 添加 BoxLabel 类
            /////// add 
            #include <opencv2/imgproc/imgproc.hpp>
            /////// add        
            等
    caffe\src\caffe\data_transformer.cpp


    \caffe\src\caffe\util\blocking_queue.cpp

        //////////////////////// add ////////////////////
        #include "caffe/data_reader.hpp"
        ////////////////////////////////////////////

        ///////////////// add ////////////////
        template class BlockingQueue<Datum*>;
        template class BlockingQueue<shared_ptr<DataReader::QueuePair> >;
        ////////////////// add ////////////

    }  // namespace caffe


    \caffe\src\caffe\util\io.cpp


    \caffe\src\caffe\layer.cpp
    \caffe\src\caffe\net.cpp
    \caffe\src\caffe\parallel.cpp  不一定要修改
    solver.cpp                      不一定要修改



