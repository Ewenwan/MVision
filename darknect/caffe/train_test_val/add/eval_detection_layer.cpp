#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/detection_loss_layer.hpp"
#include "caffe/layers/eval_detection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

class BoxData {
 public:
  int label_;
  bool difficult_;
  float score_;
  vector<float> box_;
};

bool BoxSortDecendScore(const BoxData& box1, const BoxData& box2) {
  return box1.score_ > box2.score_;
}

void ApplyNms(const vector<BoxData>& boxes, vector<int>* idxes, float threshold) {
  map<int, int> idx_map;
  for (int i = 0; i < boxes.size() - 1; ++i) {
    if (idx_map.find(i) != idx_map.end()) {
      continue;
    }
    vector<float> box1 = boxes[i].box_;
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (idx_map.find(j) != idx_map.end()) {
        continue;
      }
      vector<float> box2 = boxes[j].box_;
      float iou = Calc_iou(box1, box2);
      if (iou >= threshold) {
        idx_map[j] = 1;
      }
    }
  }
  for (int i = 0; i < boxes.size(); ++i) {
    if (idx_map.find(i) == idx_map.end()) {
      idxes->push_back(i);
    }
  }
}

template <typename Dtype>
void GetGTBox(int side, const Dtype* label_data, map<int, vector<BoxData> >* gt_boxes) {
  int locations = pow(side, 2);
  for (int i = 0; i < locations; ++i) {
    if (!label_data[locations + i]) {
      continue;
    }
    BoxData gt_box;
    bool difficult = (label_data[i] == 1);
    int label = static_cast<int>(label_data[locations * 2 + i]);
    gt_box.difficult_ = difficult;
    gt_box.label_ = label;
    gt_box.score_ = i;
    int box_index = locations * 3 + i * 4;
    for (int j = 0; j < 4; ++j) {
      gt_box.box_.push_back(label_data[box_index + j]);
    }
    if (gt_boxes->find(label) == gt_boxes->end()) {
      (*gt_boxes)[label] = vector<BoxData>(1, gt_box);
    } else {
      (*gt_boxes)[label].push_back(gt_box);
    }
  }
}

template <typename Dtype>
void GetPredBox(int side, int num_object, int num_class, const Dtype* input_data,
            map<int, vector<BoxData> >* pred_boxes, bool use_sqrt, bool constriant, 
            int score_type, float nms_threshold) {
  vector<BoxData> tmp_boxes;
  int locations = pow(side, 2);
  for (int i = 0; i < locations; ++i) {
    int pred_label = 0;
    float max_prob = input_data[i];
    for (int j = 1; j < num_class; ++j) {
      int class_index = j * locations + i;   
      if (input_data[class_index] > max_prob) {
        pred_label = j;
        max_prob = input_data[class_index];
      }
    }
    if (nms_threshold < 0) {
      if (pred_boxes->find(pred_label) == pred_boxes->end()) {
        (*pred_boxes)[pred_label] = vector<BoxData>();
      }
    }
    // LOG(INFO) << "pred_label: " << pred_label << " max_prob: " << max_prob; 
    int obj_index = num_class * locations + i;
    int coord_index = (num_class + num_object) * locations + i;
    for (int k = 0; k < num_object; ++k) {
      BoxData pred_box;
      float scale = input_data[obj_index + k * locations];
      pred_box.label_ = pred_label;
      if (score_type == 0) {
        pred_box.score_ = scale;
      } else if (score_type == 1) {
        pred_box.score_ = max_prob;
      } else {
        pred_box.score_ = scale * max_prob;
      }
      int box_index = coord_index + k * 4 * locations;
      if (!constriant) {
        pred_box.box_.push_back(input_data[box_index + 0 * locations]);
        pred_box.box_.push_back(input_data[box_index + 1 * locations]);
      } else {
        pred_box.box_.push_back((i % side + input_data[box_index + 0 * locations]) / side);
        pred_box.box_.push_back((i / side + input_data[box_index + 1 * locations]) / side);
      }
      float w = input_data[box_index + 2 * locations];
      float h = input_data[box_index + 3 * locations];
      if (use_sqrt) {
        pred_box.box_.push_back(pow(w, 2));
        pred_box.box_.push_back(pow(h, 2));
      } else {
        pred_box.box_.push_back(w);
        pred_box.box_.push_back(h);
      }
      if (nms_threshold >= 0) {
        tmp_boxes.push_back(pred_box);
      } else {
        (*pred_boxes)[pred_label].push_back(pred_box);
      }
    }
  }
  if (nms_threshold >= 0) {
    std::sort(tmp_boxes.begin(), tmp_boxes.end(), BoxSortDecendScore);
    vector<int> idxes;
    ApplyNms(tmp_boxes, &idxes, nms_threshold);
    for (int i = 0; i < idxes.size(); ++i) {
      BoxData box_data = tmp_boxes[idxes[i]];
      if (pred_boxes->find(box_data.label_) == pred_boxes->end()) {
        (*pred_boxes)[box_data.label_] = vector<BoxData>();
      }
      (*pred_boxes)[box_data.label_].push_back(box_data);
    }
  } else {
    for (std::map<int, vector<BoxData> >::iterator it = pred_boxes->begin(); it != pred_boxes->end(); ++it) {
      std::sort(it->second.begin(), it->second.end(), BoxSortDecendScore);
    }
  }
}

template <typename Dtype>
void EvalDetectionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  EvalDetectionParameter param = this->layer_param_.eval_detection_param();
  side_ = param.side();
  num_class_ = param.num_class();
  num_object_ = param.num_object();
  threshold_ = param.threshold();
  sqrt_ = param.sqrt();
  constriant_ = param.constriant();
  nms_ = param.nms();
  switch (param.score_type()) {
    case EvalDetectionParameter_ScoreType_OBJ:
      score_type_ = 0;
      break;
    case EvalDetectionParameter_ScoreType_PROB:
      score_type_ = 1;
      break;
    case EvalDetectionParameter_ScoreType_MULTIPLY:
      score_type_ = 2;
      break;
    default:
      LOG(FATAL) << "Unknow score type.";
  }
}

template <typename Dtype>
void EvalDetectionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int input_count = bottom[0]->count(1);
  int label_count = bottom[1]->count(1);
  // outputs: classes, iou, coordinates
  int tmp_input_count = side_ * side_ * (num_class_ + (1 + 4) * num_object_);
  // label: isobj, class_label, coordinates
  int tmp_label_count = side_ * side_ * (1 + 1 + 1 + 4);
  CHECK_EQ(input_count, tmp_input_count);
  CHECK_EQ(label_count, tmp_label_count);

  vector<int> top_shape(2, 1);
  top_shape[0] = bottom[0]->num();
  top_shape[1] = num_class_ + side_ * side_ * num_object_ * 4; 
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void EvalDetectionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    int input_index = i * bottom[0]->count(1);
    int true_index = i * bottom[1]->count(1);
    int top_index = i * top[0]->count(1);
    map<int, vector<BoxData> > gt_boxes;
    GetGTBox(side_, label_data + true_index, &gt_boxes);
    for (std::map<int, vector<BoxData > >::iterator it = gt_boxes.begin(); it != gt_boxes.end(); ++it) {
      int label = it->first;
      vector<BoxData>& g_boxes = it->second;
      for (int j = 0; j < g_boxes.size(); ++j) {
        if (!g_boxes[j].difficult_) {
          top_data[top_index + label] += 1;
        }
      }
    }
    map<int, vector<BoxData> > pred_boxes;
    GetPredBox(side_, num_object_, num_class_, input_data + input_index, &pred_boxes, sqrt_, constriant_, score_type_, nms_);
    int index = top_index + num_class_;
    int pred_count(0);
    for (std::map<int, vector<BoxData> >::iterator it = pred_boxes.begin(); it != pred_boxes.end(); ++it) {
      int label = it->first;
      vector<BoxData>& p_boxes = it->second;
      if (gt_boxes.find(label) == gt_boxes.end()) {
        for (int b = 0; b < p_boxes.size(); ++b) {
          top_data[index + pred_count * 4 + 0] = p_boxes[b].label_;
          top_data[index + pred_count * 4 + 1] = p_boxes[b].score_;
          top_data[index + pred_count * 4 + 2] = 0;
          top_data[index + pred_count * 4 + 3] = 1;
          ++pred_count;
        }
        continue;
      } 
      vector<BoxData>& g_boxes = gt_boxes[label];
      vector<bool> records(g_boxes.size(), false);
      for (int k = 0; k < p_boxes.size(); ++k) {
        top_data[index + pred_count * 4 + 0] = p_boxes[k].label_;
        top_data[index + pred_count * 4 + 1] = p_boxes[k].score_;
        float max_iou(-1);
        int idx(-1);
        for (int g = 0; g < g_boxes.size(); ++g) {
          float iou = Calc_iou(p_boxes[k].box_, g_boxes[g].box_);
          if (iou > max_iou) {
            max_iou = iou;
            idx = g;
          }
        }
        if (max_iou >= threshold_) {
          if (!g_boxes[idx].difficult_) {
            if (!records[idx]) {
              records[idx] = true;
              top_data[index + pred_count * 4 + 2] = 1;
              top_data[index + pred_count * 4 + 3] = 0;
            } else {
              top_data[index + pred_count * 4 + 2] = 0;
              top_data[index + pred_count * 4 + 3] = 1;
            }
          }
        } else {
          top_data[index + pred_count * 4 + 2] = 0;
          top_data[index + pred_count * 4 + 3] = 1;
        }
        ++pred_count;
      }
    }
  }
}

INSTANTIATE_CLASS(EvalDetectionLayer);
REGISTER_LAYER_CLASS(EvalDetection);

}  // namespace caffe
