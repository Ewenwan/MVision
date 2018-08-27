# 卷积层

```c
// 层类初始化函数==============
template <typename Dtype>
void ConvolutionRistrettoLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, // 上层激活输入 作为本层输入  输入通道数*输入h*输入w
      const vector<Blob<Dtype>*>& top)    // 本层激活输出               输出通道数*输出h*输出w
{
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();//本层卷积参数=====

// 配置输入=====
  this->force_nd_im2col_ = conv_param.force_nd_im2col();// ?? 强制使用n维通用卷积
  
  this->channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());// 输入的第几个轴是通道
  
  const int first_spatial_axis = this->channel_axis_ + 1;// 第一个空间轴 id
  
  const int num_axes = bottom[0]->num_axes();// 轴总数量
  
  this->num_spatial_axes_ = num_axes - first_spatial_axis;// 空间轴 总数
  
  CHECK_GE(this->num_spatial_axes_, 0);//  >= 0
  
  vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);// 输入层 形状 维度数量=空间轴 总数+1(通道数量)
  
  vector<int> spatial_dim_blob_shape(1, std::max(this->num_spatial_axes_, 1));// 空间维度 

// 配置 卷积核=====
  // Setup filter kernel dimensions (kernel_shape_).
  this->kernel_shape_.Reshape(spatial_dim_blob_shape);// 卷积核形状维度 同 输入特征的空间形状维度
  
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();// 卷积核 形状数据===== 3*3 卷积核
  
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) 
  {    // 以 kernel_h & kernel_w给定 卷积参数 =========
    CHECK_EQ(this->num_spatial_axes_, 2)// 空间维度数量 = 2，  2D卷积
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())// 网络参数中 kernel_size 和 kernel_h & kernel_w  只能有一个
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();// 3*3 
    kernel_shape_data[1] = conv_param.kernel_w();
  } 
  else // 以 kernel_size 指定 卷积参数 ================
  {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == this->num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < this->num_spatial_axes_; ++i) 
	  {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  
  
  for (int i = 0; i < this->num_spatial_axes_; ++i) 
  {// 卷积核 尺寸 大于0
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  
// 配置 步长stride 参数============================ 
  // Setup stride dimensions (stride_).
  this->stride_.Reshape(spatial_dim_blob_shape);// 步长 stride_ 的 维度数量和 输入特征空间维度一致
  int* stride_data = this->stride_.mutable_cpu_data();// 步长 stride_ 尺寸数据 1*1
  
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) 
  {// 以 stride_h & stride_w 给定 步长stride_参数
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();// 1*1
    stride_data[1] = conv_param.stride_w();// 
  } 
  else
  {// 以 stride 给定 步长stride_ 参数
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == this->num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }

// 配置填充 padding 参数=============================
  // Setup pad dimensions (pad_).
  this->pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = this->pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) 
  {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } 
  else 
  {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == this->num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }

// 配置空洞卷积参数================================
  // Setup dilation dimensions (dilation_).
  this->dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = this->dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == this->num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << this->num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < this->num_spatial_axes_; ++i) 
  {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }

// 是不是1x1卷积========
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  this->is_1x1_ = true;
  for (int i = 0; i < this->num_spatial_axes_; ++i) 
  {
    this->is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!this->is_1x1_) { break; }
  }

// 配置 输出特征尺寸 ========================
  // Configure output channels and groups.
  this->channels_ = bottom[0]->shape(this->channel_axis_);// 输入特征 通道
  this->num_output_ = this->layer_param_.convolution_param().num_output();// 输出特征 通道数量
  CHECK_GT(this->num_output_, 0);
  this->group_ = this->layer_param_.convolution_param().group();// 分组卷积信息，输入通道分组数量
  CHECK_EQ(this->channels_ % this->group_, 0); // 通道数量都要是 分组数量的整数倍
  CHECK_EQ(this->num_output_ % this->group_, 0)
      << "Number of output should be multiples of group.";
  if (this->reverse_dimensions()) // 反卷积
  {
    this->conv_out_channels_ = this->channels_;//  反卷积 和正常卷积相反 
    this->conv_in_channels_ = this->num_output_;// 
  } 
  else 
  {
    this->conv_out_channels_ = this->num_output_;// 卷积核个数 决定 输出通道数量
    this->conv_in_channels_ = this->channels_;   // 卷积通道数量
  }
  
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights    y = x*w + b  卷积权重w
  // - blobs_[1] holds the biases (optional) 偏置b 可选
  
  // 卷积权重w 数量 和 通道 尺寸 参数 
  vector<int> weight_shape(2);
  weight_shape[0] = this->conv_out_channels_; //  卷积核个数 决定 输出通道数量
  weight_shape[1] = this->conv_in_channels_ / this->group_;// 卷积核通道 考虑分组
  
  for (int i = 0; i < this->num_spatial_axes_; ++i) 
  {
    weight_shape.push_back(kernel_shape_data[i]);// 后面是 卷积核尺寸大小     N*C*W*H
  }
  
  this->bias_term_ = this->layer_param_.convolution_param().bias_term();
  
  vector<int> bias_shape(this->bias_term_, this->num_output_);// 每个输出通道 有一个偏置 参数
  
  // 网络权重已经存在，只需检测维度是否一致=====================
  if (this->blobs_.size() > 0) 
  {
    CHECK_EQ(1 + this->bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    
    if (weight_shape != this->blobs_[0]->shape()) // 权重w  ==== blobs_[0]
	{
      Blob<Dtype> weight_shaped_blob(weight_shape);// 权重w 
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (this->bias_term_ && bias_shape != this->blobs_[1]->shape()) // 偏置b ===== blobs_[1]
	{
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
	
    LOG(INFO) << "Skipping parameter initialization";
  } 
  
  else // 网络初始化时====需要使用 初始化参数填充卷积核权重============
  {
    if (this->bias_term_) 
	{
      this->blobs_.resize(2);//  blobs_[0] 存储 权重w  ； blobs_[1] 存储 偏置b 
    } 
	else 
	{
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));// 使用 权重初始化参数 
    
    weight_filler->Fill(this->blobs_[0].get());// 类似高斯分布 数据 填充 权重w=============================
    // If necessary, initialize and fill the biases.
    if (this->bias_term_) 
    {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());// 类似高斯分布 数据 填充 偏置b=============================
    }
  }
  
  
  this->kernel_dim_ = this->blobs_[0]->count(1);// 卷积核尺寸大小 1之后  N*C*W*H 卷积核的维度 = 通道C*卷积核的W*卷积核的H, N为卷积核数量
  this->weight_offset_ =
      this->conv_out_channels_ * this->kernel_dim_ / this->group_;// 使用卷积组用到的 N*C*W*H /2  分组，每组分到的卷积参数数量
	  
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  
// 量化参数部分===================================
  // Prepare quantized weights
  this->weights_quantized_.resize(2);// 权重w 和 偏置b 的量化数据
  this->weights_quantized_[0].reset(new Blob<Dtype>(weight_shape));  // 量化的  权重w
  if (this->bias_term_) 
  {
      this->weights_quantized_[1].reset(new Blob<Dtype>(bias_shape));// 量化的  偏置b
  }
}

```
