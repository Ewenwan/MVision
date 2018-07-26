
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
/////多 map ////////
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
////////////////多 boost/variant /////
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

//////// 少 #include <unistd.h>
//////////////////////////////////

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

// 图像格式 灰度图/彩色图
DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
// 是否打乱数据
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
// 转换成的数据库格式 lmdb/leveldb
DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb} for storing the result");
// 标注数据的格式 分类/检测/实例分割
DEFINE_string(anno_type, "classification",
    "The type of annotation {classification, detection}.");
// 标注文件格式
DEFINE_string(label_type, "xml",
    "The type of annotation file format.");
// 标注文件内 标注name 到 真实label id 
DEFINE_string(label_map_file, "",
    "A file with LabelMap protobuf message.");
	
DEFINE_bool(check_label, false,
    "When this option is on, check that there is no duplicated name/label.");
DEFINE_int32(min_dim, 0,
    "Minimum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(max_dim, 0,
    "Maximum dimension images are resized to (keep same aspect ratio)");
// 变形后的尺寸
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
// 图片格式 'png','jpg',...
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
// glog 日志等级
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images and annotations to the "
        "leveldb/lmdb format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
// 解析命令行参数
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_annoset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;
  const string anno_type = FLAGS_anno_type;
  AnnotatedDatum_AnnotationType type;
  const string label_type = FLAGS_label_type;
  const string label_map_file = FLAGS_label_map_file;
  const bool check_label = FLAGS_check_label;
  std::map<std::string, int> name_to_label;

  std::ifstream infile(argv[2]);// 数据集列表 文件
  // 可变配对数据 std::pair<std::string, boost::variant<int, std::string> >
  // string : string      目标检测  图片路径+标注文件路径
  // 或者 string : int    图像分类  图片路径+类别id
  std::vector<std::pair<std::string, boost::variant<int, std::string> > > lines;
  std::string filename;
  int label;
  std::string labelname;
  if (anno_type == "classification") 
  {
    while (infile >> filename >> label) {
      // 或者 string : int    图像分类  图片路径+类别id
      lines.push_back(std::make_pair(filename, label));
    }
  } 
  else if (anno_type == "detection") 
  {
    type = AnnotatedDatum_AnnotationType_BBOX;
    LabelMap label_map;
    CHECK(ReadProtoFromTextFile(label_map_file, &label_map))// 读入 
        << "Failed to read label map file.";
    CHECK(MapNameToLabel(label_map, check_label, &name_to_label))
        << "Failed to convert name to label.";
    
    while (infile >> filename >> labelname) {
     // string : string      目标检测  图片路径+标注文件路径
      lines.push_back(std::make_pair(filename, labelname));
    }
  }
  
////// 打乱数据=================================
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int min_dim = std::max<int>(0, FLAGS_min_dim);
  int max_dim = std::max<int>(0, FLAGS_max_dim);
  
  // 尺寸变形 固定到网络输入大小==========
  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB  创建数据库文件
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  
// Datum datum;================================
  AnnotatedDatum anno_datum;
  Datum* datum = anno_datum.mutable_datum();// 分类数据 指针
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  // 
 // 遍历每一张数据===========================================
  for (int line_id = 0; line_id < lines.size(); ++line_id) 
  {
    bool status = true;
    std::string enc = encode_type;
	
// 图片编码格式=====================
	if (encoded && !enc.size()) 
	{
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    filename = root_folder + lines[line_id].first;// 图片文件地址
 // 转换 分类数据
    if (anno_type == "classification") 
	{
      label = boost::get<int>(lines[line_id].second);// 类别id
      status = ReadImageToDatum(filename, // 图片路径
	                            label,    // 类别id  int 
								resize_height, // 尺寸变形 固定到网络输入大小
								resize_width,  // 
                                min_dim, max_dim, 
								is_color, // 彩色图像
								enc,      // 编码格式
								datum);   // 转换到的数据
    } 
// 转换检测数据
// 图像数据
// datum = anno_datum->mutable_datum()
//  datum->set_data();                      // 所有8位像素数据
//  datum->set_channels(cv_img.channels()); // 通道数量
//  datum->set_height(cv_img.rows);         // 行
//  datum->set_width(cv_img.cols);          // 列
//  datum->set_encoded(true);               // 编码?
// 标签
// anno_datum->mutable_annotation_group(g)->set_group_label(label)   标签
//                                        ->add_annotation()->mutable_bbox()  边框数据
	else if (anno_type == "detection") 
	{
	// 边框类别标注 文件路径
      labelname = root_folder + boost::get<std::string>(lines[line_id].second);
	  
      status = ReadRichImageToAnnotatedDatum(
	      filename,     // 图片路径
	      labelname,    // 边框类别标注 文件路径
		  resize_height,// 尺寸变形 固定到网络输入大小
          resize_width, 
		  min_dim, 
		  max_dim, 
		  is_color, // 彩色图像
		  enc,      // 编码格式
		  type,     // AnnotatedDatum_AnnotationType_BBOX
		  label_type,// 标签文件格式  txt/xml/json
          name_to_label, // 标注文件 name标签 : class_id
		  &anno_datum);  // 转换到的数据
	   
      anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
    }
	
    if (status == false) 
	{
      LOG(WARNING) << "Failed to read " << lines[line_id].first;
      continue;
    }
// 尺寸大小==========================
    if (check_size)
	{
      if (!data_size_initialized) {
        data_size = datum->channels() * datum->height() * datum->width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum->data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
// 头  每一行id_图片路径 ===================
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(anno_datum.SerializeToString(&out));
    txn->Put(key_str, out);
	
/////////////// 每转换 1000张数据 显示一次信息 并保存数据
    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
///////// 最后一些图片数据 不一定有1000张
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
