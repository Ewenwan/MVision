#include <fcntl.h>
////////////////// 添加的头文件
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>// boost库之ptree解析xml
#include <boost/property_tree/xml_parser.hpp>
/////////////////////
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {
//////////////////////////////////////////// 添加声明///
using namespace boost::property_tree;
//////////////////////////////////////////
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV

////////////////////////////////////// 6 参数 ////
// 变形到 指定 大小
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color,
    int* ori_w, int* ori_h) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  *ori_w = cv_img_origin.cols;
  *ori_h = cv_img_origin.rows;
  if (height > 0 && width > 0) {
    // 变形
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}
/////////////////////////////////////////////

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p+1) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}
// 添加转换 边框数据的接口函数
bool ReadBoxDataToDatum(const string& filename, const string& annoname,
    const map<string, int>& label_map, const int height, const int width, 
    const bool is_color, const std::string & encoding, Datum* datum) {
  int ori_w, ori_h;
  // 图片 变形到指定格式
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color, &ori_w, &ori_h);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, annoname, label_map, ori_w, ori_h, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_encoded(true);
      // read xml anno data
      ParseXmlToDatum(annoname, label_map, ori_w, ori_h, datum);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    // read xml anno data
    ParseXmlToDatum(annoname, label_map, ori_w, ori_h, datum);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV
// 字符串标签 转换成 整数标签
int name_to_label(const string& name, const map<string, int>& label_map) {
  map<string, int>::const_iterator it = label_map.find(name);//  map 查找
  if (it == label_map.end()) 
    return -1;
  else
    return it->second;// 整数键值
}
// 转换标注文件 
void ParseXmlToDatum(const string& annoname, const map<string, int>& label_map,
    int ori_w, int ori_h, Datum* datum) {
  ptree pt;
  read_xml(annoname, pt);
  int width(0), height(0);
  try {
    height = pt.get<int>("annotation.size.height");
    width = pt.get<int>("annotation.size.width");
    CHECK_EQ(ori_w, width);// 在 范围内
    CHECK_EQ(ori_h, height);
  } catch (const ptree_error &e) {
    LOG(WARNING) << "When paring " << annoname << ": " << e.what();
  }
  datum->clear_float_data();
  BOOST_FOREACH(ptree::value_type &v1, pt.get_child("annotation")) {
    if (v1.first == "object") {
      ptree object = v1.second;
      int label(-1);// 一个类别标签
      vector<float> box(4, 0);// 四个边框参数
      int difficult(0);
      BOOST_FOREACH(ptree::value_type &v2, object.get_child("")) {
        ptree pt2 = v2.second;
        if (v2.first == "name") {
          string name = pt2.data();// 字符串 标签
          // map name to label
          label = name_to_label(name, label_map);// 字符串标签 转换成 整数标签
          if (label < 0) {
            LOG(FATAL) << "Anno file " << annoname << " -> unknown name: " << name;
          }
        } else if (v2.first == "bndbox") {
          int xmin = pt2.get("xmin", 0);// 左下角 坐标 (xmin,ymin)
          int ymin = pt2.get("ymin", 0);
          int xmax = pt2.get("xmax", 0);// 右上角 坐标 (xmax,ymax)
          int ymax = pt2.get("ymax", 0);
           // 判断标签 合理性
          LOG_IF(WARNING, xmin < 0 || xmin > ori_w) << annoname << 
              " bounding box exceeds image boundary";
          LOG_IF(WARNING, xmax < 0 || xmax > ori_w) << annoname << 
              " bounding box exceeds image boundary";
          LOG_IF(WARNING, ymin < 0 || ymin > ori_h) << annoname << 
              " bounding box exceeds image boundary";
          LOG_IF(WARNING, ymax < 0 || ymax > ori_h) << annoname << 
              " bounding box exceeds image boundary";
          LOG_IF(WARNING, xmin > xmax) << annoname << 
              " bounding box exceeds image boundary";
          LOG_IF(WARNING, ymin > ymax) << annoname << 
              " bounding box exceeds image boundary";
             // 边框中心点 坐标 所占 原图片大小的位置比例 
          box[0] = float(xmin + (xmax - xmin) / 2.) / ori_w;
          box[1] = float(ymin + (ymax - ymin) / 2.) / ori_h;
             // 边框 尺寸 占据 原图片大小的比例
          box[2] = float(xmax - xmin) / ori_w;
          box[3] = float(ymax - ymin) / ori_h;
        } else if (v2.first == "difficult") {
          difficult = atoi(pt2.data().c_str());//  
        }
      }
      CHECK_GE(label, 0) << "label must start at 0";
      datum->add_float_data(float(label));// 标签
      datum->add_float_data(float(difficult));
      for (int i = 0; i < 4; ++i) {
        datum->add_float_data(box[i]);// 边框数据
      }
    }
  }
}

bool ReadFileToDatum(const string& filename, const string& annoname,
      const map<string, int>& label_map, int ori_w, int ori_h, Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_encoded(true);
    ParseXmlToDatum(annoname, label_map, ori_w, ori_h, datum);
    return true;
  } else {
    return false;
  }
}

////////////////////////////////////


bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}
#endif// USE_OPENCV

}// namespace caffe
