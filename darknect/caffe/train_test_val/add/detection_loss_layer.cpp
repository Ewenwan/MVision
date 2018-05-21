#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/detection_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
  Dtype left = std::max(x1 - w1 / 2, x2 - w2 / 2);
  Dtype right = std::min(x1 + w1 / 2, x2 + w2 / 2);
  return right - left;
}

template <typename Dtype>
Dtype Calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth) {
  Dtype w = Overlap(box[0], box[2], truth[0], truth[2]);
  Dtype h = Overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;
  Dtype inter_area = w * h;
  Dtype union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area / union_area;
}

template <typename Dtype>
Dtype Calc_rmse(const vector<Dtype>& box, const vector<Dtype>& truth) {
  return sqrt(pow(box[0]-truth[0], 2) +
              pow(box[1]-truth[1], 2) +
              pow(box[2]-truth[2], 2) +
              pow(box[3]-truth[3], 2));
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  DetectionLossParameter param = this->layer_param_.detection_loss_param();
  side_ = param.side();
  num_class_ = param.num_class();
  num_object_ = param.num_object();
  sqrt_ = param.sqrt();
  constriant_ = param.constriant();
  object_scale_ = param.object_scale();
  noobject_scale_ = param.noobject_scale();
  class_scale_ = param.class_scale();
  coord_scale_ = param.coord_scale();
  
  int input_count = bottom[0]->count(1);
  int label_count = bottom[1]->count(1);
  // outputs: classes, iou, coordinates
  int tmp_input_count = side_ * side_ * (num_class_ + (1 + 4) * num_object_);
  // label: isobj, class_label, coordinates
  int tmp_label_count = side_ * side_ * (1 + 1 + 1 + 4);
  CHECK_EQ(input_count, tmp_input_count);
  CHECK_EQ(label_count, tmp_label_count);
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* diff = diff_.mutable_cpu_data();
  Dtype loss(0.0), class_loss(0.0), noobj_loss(0.0), obj_loss(0.0), coord_loss(0.0), area_loss(0.0);
  Dtype avg_iou(0.0), avg_obj(0.0), avg_cls(0.0), avg_pos_cls(0.0), avg_no_obj(0.0);
  Dtype obj_count(0);
  int locations = pow(side_, 2);
  caffe_set(diff_.count(), Dtype(0.), diff);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    int index = i * bottom[0]->count(1);
    int true_index = i * bottom[1]->count(1);
    for (int j = 0; j < locations; ++j) {
      for (int k = 0; k < num_object_; ++k) {
        int p_index = index + num_class_ * locations + k * locations + j;
        noobj_loss += noobject_scale_ * pow(input_data[p_index] - 0, 2);
        diff[p_index] = noobject_scale_ * (input_data[p_index] - 0);
        avg_no_obj += input_data[p_index];
      }
      bool isobj = label_data[true_index + locations + j];
      if (!isobj) {
        continue;
      }
      obj_count += 1;
      int label = static_cast<int>(label_data[true_index + locations * 2 + j]);
      CHECK_GE(label, 0) << "label start at 0";
      CHECK_LT(label, num_class_) << "label must below num_class";
      for (int c = 0; c < num_class_; ++c) {
        int class_index = index + c * locations + j;
        Dtype target = Dtype(c == label);
        avg_cls += input_data[class_index];
        if (c == label)
          avg_pos_cls += input_data[class_index]; 
        class_loss += class_scale_ * pow(input_data[class_index] - target, 2);
        diff[class_index] = class_scale_ * (input_data[class_index] - target);
      }
      const Dtype* true_box_pt = label_data + true_index + locations * 3 + j * 4;
      vector<Dtype> true_box(true_box_pt, true_box_pt + 4);
      const Dtype* box_pt = input_data + index + (num_class_+num_object_)*locations + j;
      Dtype best_iou = 0.;
      Dtype best_rmse = 20.;
      int best_index = 0;
      for (int k = 0; k < num_object_; ++k) {
        vector<Dtype> box;
        box.push_back(*(box_pt + (k * 4 + 0) * locations));
        box.push_back(*(box_pt + (k * 4 + 1) * locations));
        box.push_back(*(box_pt + (k * 4 + 2) * locations));
        box.push_back(*(box_pt + (k * 4 + 3) * locations));
        if (constriant_) {
          box[0] = (j % side_ + box[0]) / side_;
          box[1] = (j / side_ + box[1]) / side_;
        }
        if (sqrt_) {
          box[2] = pow(box[2], 2);
          box[3] = pow(box[3], 2);
        }
        Dtype iou = Calc_iou(box, true_box);
        Dtype rmse = Calc_rmse(box, true_box);
        if (best_iou > 0 || iou > 0) {
          if (iou > best_iou) {
            best_iou = iou;
            best_index = k;
          }
        } else {
          if (rmse < best_rmse) {
            best_rmse = rmse;
            best_index = k;
          }
        }
      }

      CHECK_GE(best_index, 0) << "best_index must >= 0";
      avg_iou += best_iou;
      int p_index = index + num_class_ * locations + best_index * locations + j;
      noobj_loss -= noobject_scale_ * pow(input_data[p_index], 2);
      obj_loss += object_scale_ * pow(input_data[p_index] - 1., 2);
      avg_no_obj -= input_data[p_index];
      avg_obj += input_data[p_index];
      // rescore
      diff[p_index] = object_scale_ * (input_data[p_index] - best_iou);
      int box_index = index + (num_class_ + num_object_ + best_index * 4) * locations + j;
      vector<Dtype> best_box;
      best_box.push_back(input_data[box_index + 0 * locations]);
      best_box.push_back(input_data[box_index + 1 * locations]);
      best_box.push_back(input_data[box_index + 2 * locations]);
      best_box.push_back(input_data[box_index + 3 * locations]);


      if (constriant_) {
        true_box[0] = true_box[0] * side_ - Dtype(j % side_);
        true_box[1] = true_box[1] * side_ - Dtype(j / side_);
      }

      if (sqrt_) {
        true_box[2] = sqrt(true_box[2]);
        true_box[3] = sqrt(true_box[3]);
      }

      for (int o = 0; o < 4; ++o) {
        diff[box_index + o * locations] = coord_scale_ * (best_box[o] - true_box[o]);
      }

      coord_loss += coord_scale_ * pow(best_box[0] - true_box[0], 2);
      coord_loss += coord_scale_ * pow(best_box[1] - true_box[1], 2);
      area_loss += coord_scale_ * pow(best_box[2] - true_box[2], 2);
      area_loss += coord_scale_ * pow(best_box[3] - true_box[3], 2);
    }
  }
  class_loss /= obj_count;
  coord_loss /= obj_count;
  area_loss /= obj_count;
  obj_loss /= obj_count;
  noobj_loss /= (locations * num_object_ * bottom[0]->num() - obj_count);

  avg_iou /= obj_count;
  avg_obj /= obj_count;
  avg_no_obj /= (locations * num_object_ * bottom[0]->num() - obj_count);
  avg_cls /= obj_count;
  avg_pos_cls /= obj_count;

  loss = class_loss + coord_loss + area_loss + obj_loss + noobj_loss;
  obj_count /= bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;

  // LOG(INFO) << "average objects: " << obj_count;
  LOG(INFO) << "loss: " << loss << " class_loss: " << class_loss << " obj_loss: " 
        << obj_loss << " noobj_loss: " << noobj_loss << " coord_loss: " << coord_loss
        << " area_loss: " << area_loss;
  LOG(INFO) << "avg_iou: " << avg_iou << " avg_obj: " << avg_obj << " avg_no_obj: "
        << avg_no_obj << " avg_cls: " << avg_cls << " avg_pos_cls: " << avg_pos_cls;
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype sign(1.);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_cpu_axpby(
        bottom[0]->count(),
        alpha,
        diff_.cpu_data(),
        Dtype(0),
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(DetectionLossLayer);
#endif

INSTANTIATE_CLASS(DetectionLossLayer);
REGISTER_LAYER_CLASS(DetectionLoss);

}  // namespace caffe
