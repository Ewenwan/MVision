
// yolov2 一拆四层=================================
//////////////////////////////////////////////
// 大尺寸图 拆分成 4块小的尺寸图
// 再 和 经卷积下采样的小尺寸图 concat结合===============
/////////////////////////////////////////////////// 
#include "caffe/layers/reorg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template<typename Dtype>
    void ReorgLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom,// 输入大尺寸  N* channel * H * W
    const vector<Blob<Dtype> *> &top)   // 输出小尺寸  N* (4*channel)* (H/2) * (W/2)
	{
        CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
                    "allow in-place computation.";
        ReorgParameter reorg_param = this->layer_param_.reorg_param();// stride =2 间隔采用步长
        CHECK_EQ(reorg_param.has_stride(), true) << this->type() << " Layer needs stride param.";
        reverse_ = reorg_param.reverse();// 标志  小尺寸 -> 大尺寸 / 大尺寸 -> 小尺寸
        stride_ = reorg_param.stride();  // stride =2 间隔采用步长 
        channels_ = bottom[0]->channels();// 输入 数据通道数量 
        height_ = bottom[0]->height();// 特征图尺寸 
        width_ = bottom[0]->width();
        batch_num_ = bottom[0]->num();// 图片数量 N

        diff_.Reshape(batch_num_, channels_, height_, width_);

        if (reverse_) 
		{// 小尺寸 -> 大尺寸
            reorged_channels_ = channels_ / (stride_ * stride_);// 
            reorged_width_ = width_ * stride_;
            reorged_height_ = height_ * stride_;
        } 
		else 
		{
            reorged_channels_ = channels_ * stride_ * stride_;// 通道数量 扩展成4倍数
            reorged_height_ = height_ / stride_;// 特征图尺寸 降低为原来的 一半
            reorged_width_ = width_ / stride_;
        }
    }
	
// 变形===========================================================
    template<typename Dtype>
    void ReorgLayer<Dtype>::Reshape(
	    const vector<Blob<Dtype> *> &bottom,
        const vector<Blob<Dtype> *> &top) 
		
{ // 输入大尺寸  N* channel * H * W  ->  输出小尺寸  N* (4*channel)* (H/2) * (W/2)
        top[0]->Reshape(batch_num_, 
                        reorged_channels_,
                        reorged_height_, 
                        reorged_width_);
 }
// 网络前传，特征图 下采样到4倍通道数 =======================================================
    template<typename Dtype>
    void ReorgLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
		
        const Dtype *bottom_data = bottom[0]->cpu_data();// 输入数据
		
        Dtype *top_data = top[0]->mutable_cpu_data();    // 输出数据
		//  在 reorg_layer.hpp 头文件中
        reorg_cpu(bottom_data, // 输入大尺寸  N* channel * H * W
		          width_,      // W
				  height_,     // H
                  channels_,   // channel 
				  batch_num_,  // N
				  stride_,     // stride =2 间隔采用步长 
				  reverse_,    // 是否反向采样
				  top_data);   // 输出小尺寸  N* (4*channel)* (M/2) * (N/2)
    }
// 网络 反传， 梯度图 上采样到1/4通道数======================================================
    template<typename Dtype>
    void ReorgLayer<Dtype>::Backward_cpu(
        const vector<Blob<Dtype> *> &top,    //    变形后输出
        const vector<bool> &propagate_down,
        const vector<Blob<Dtype> *> &bottom) // 底部输入梯度
{
        if(!propagate_down[0]){
            return;
        }
		// 对梯度 也 进行变形=======================  
        //const Dtype *top_diff = top[0]->cpu_diff();
        const Dtype *top_diff = diff_.mutable_cpu_diff();// 新建 cpu梯度数据
        Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
        reorg_cpu(top_diff,   // 变大的输出数据
		          width_, 
				  height_,
                  channels_, 
				  batch_num_, 
				  stride_, 
				  !reverse_,   // 梯度反过来  小图 组合 成 大图 通道数量 减少 
				  bottom_diff);// 底部输入梯度
    }
	
#ifdef CPU_ONLY
STUB_GPU(ReorgLayer);
#endif
    INSTANTIATE_CLASS(ReorgLayer);

    REGISTER_LAYER_CLASS(Reorg);

}  // namespace caffe
