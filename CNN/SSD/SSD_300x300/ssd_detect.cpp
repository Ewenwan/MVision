// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// 模型文件格式 model_file  .prototxt 
// 权重文件格式 weights_file   .caffemodel
// 图像列表文件
//    folder/img1.JPEG
//    folder/img2.JPEG
// 视频列表文件
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

// 检测器类
class Detector {
 public:
 // 类构造函数
  Detector(const string& model_file,//模型文件
           const string& weights_file,//权重文件
           const string& mean_file,//图像三通道均值
           const string& mean_value);
  // 检测 输出为二维数组
  std::vector<vector<float> > Detect(const cv::Mat& img);

 private:
 // 均值
  void SetMean(const string& mean_file, const string& mean_value);
// 改变图像大小
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
// 处理图像
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;//caffe网络模型
  cv::Size input_geometry_;// 输入图像大小
  int num_channels_;//通道数量
  cv::Mat mean_;//均值
};

// 类构造函数
Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);//cpu模式
#else
  Caffe::set_mode(Caffe::GPU);//gpu模式
#endif

  // 载入模型和权重文件
  net_.reset(new Net<float>(model_file, TEST));// 模型框架
  net_->CopyTrainedLayersFrom(weights_file);// 参数权重
  // 模型输入输出数量
  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
  
  // 模型最开始的输入 也就是输入数据的形状 1*3*300*300
  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();//图像通道数量
  CHECK(num_channels_ == 3 || num_channels_ == 1)//彩色图或者灰度图
    << "Input layer should have 1 or 3 channels.";
  // 模型输入图像的 宽高尺寸 
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

// 网络检测输出 二维数组输出结果
std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();
  
// 模型输入的数据块
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  
  // 预处理图像 通道 大小 浮点数类型 去均值
  Preprocess(img, &input_channels);
  
  // 模型前向传播
  net_->Forward();

  // 模型结果输出
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();// 总的检测框数量
  vector<vector<float> > detections;// 二维数组 所以的检测结果
  for (int k = 0; k < num_det; ++k) {// 每一个 检测框
    if (result[0] == -1) {// 跳过异常的检测值
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);//单个检测结果
    // image_id 标签索引  可信度  四个坐标
    // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

// voc 图像通道 均值
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

// 图像变形到模型输入的指定大小
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  // 模型输入图像的指定大小
  Blob<float>* input_layer = net_->input_blobs()[0];
  int width = input_layer->width();
  int height = input_layer->height();
  
  float* input_data = input_layer->mutable_cpu_data();//数据指针
  for (int i = 0; i < input_layer->channels(); ++i) {// 每一个通道
    cv::Mat channel(height, width, CV_32FC1, input_data);//形成一通道
    input_channels->push_back(channel);
    input_data += width * height;// 数据指针+一个通道的像素数量
  }
}
// 图像预处理
void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
 
// 【1】 转换图像通道 符合模型输入通道格式
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    // 模型输入为1通道,而3通道的图片转换成1通道的灰度图
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    // 模型输入为1通道,而4通道的图片转换成1通道的灰度图
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    // 模型输入为3通道,而4通道的图片rgba 转换成3通道的rgb
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    // 模型输入为3通道,而1通道的灰度图片 转换成3通道的rgb
  else
    sample = img;

// 【2】转换图像大小到模型输入大小 input_geometry_
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

// 【3】图片数据存储类型转换到float
  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

// 【4】各个通道减去均值
  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

// 【5】通道分割到 模型输入的数据块
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.01,
    "Only store detections with score higher than the threshold.");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }

  const string& model_file = argv[1];// 模型文件
  const string& weights_file = argv[2];// 模型权重文件
  const string& mean_file = FLAGS_mean_file;//均值文件
  const string& mean_value = FLAGS_mean_value;//
  const string& file_type = FLAGS_file_type;// 文件类型 图片/视频/
  const string& out_file = FLAGS_out_file;
  const float confidence_threshold = FLAGS_confidence_threshold;// 置信度阈值

  // 初始化一个检测器对象
  Detector detector(model_file, weights_file, mean_file, mean_value);

  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ofstream outfile;
  if (!out_file.empty()) {
    outfile.open(out_file.c_str());
    if (outfile.good()) {
      buf = outfile.rdbuf();
    }
  }
  std::ostream out(buf);

  // Process image one by one.
  std::ifstream infile(argv[3]);// 测试图片目录列表文件
  std::string file;
  while (infile >> file) {//每一行是一个图像的地址
  // 图像
    if (file_type == "image") {
      cv::Mat img = cv::imread(file, -1);
      CHECK(!img.empty()) << "Unable to decode image " << file;
      // 获取该图像的 模型检测结果
      std::vector<vector<float> > detections = detector.Detect(img);

      /* Print the detection results. */
      for (int i = 0; i < detections.size(); ++i) {// 遍历每一个检测框
        const vector<float>& d = detections[i];// 每一个检测框
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);//大小为7
        const float score = d[2];//置信度得分
        if (score >= confidence_threshold) {
          out << file << " ";
          out << static_cast<int>(d[1]) << " ";// label
          out << score << " ";
          out << static_cast<int>(d[3] * img.cols) << " ";
          out << static_cast<int>(d[4] * img.rows) << " ";
          out << static_cast<int>(d[5] * img.cols) << " ";
          out << static_cast<int>(d[6] * img.rows) << std::endl;
        }
      }
    }
  // 视频文件
    else if (file_type == "video") {
      cv::VideoCapture cap(file);// 打开视频
      if (!cap.isOpened()) {
        LOG(FATAL) << "Failed to open video: " << file;
      }
      cv::Mat img;
      int frame_count = 0;
      while (true) {
        bool success = cap.read(img);//获取每一帧图像
        if (!success) {
          LOG(INFO) << "Process " << frame_count << " frames from " << file;
          break;
        }
        CHECK(!img.empty()) << "Error when read frame";
        //  模型检测输出结果
        std::vector<vector<float> > detections = detector.Detect(img);

        /* Print the detection results. */
        for (int i = 0; i < detections.size(); ++i) {
          const vector<float>& d = detections[i];
          // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
          CHECK_EQ(d.size(), 7);
          const float score = d[2];
          if (score >= confidence_threshold) {
            out << file << "_";
            //  输出格式
            out << std::setfill('0') << std::setw(6) << frame_count << " ";
            out << static_cast<int>(d[1]) << " ";
            out << score << " ";
            out << static_cast<int>(d[3] * img.cols) << " ";
            out << static_cast<int>(d[4] * img.rows) << " ";
            out << static_cast<int>(d[5] * img.cols) << " ";
            out << static_cast<int>(d[6] * img.rows) << std::endl;
          }
        }
        ++frame_count;
      }
      if (cap.isOpened()) {
        cap.release();
      }
    } else {
      LOG(FATAL) << "Unknown file_type: " << file_type;
    }
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
