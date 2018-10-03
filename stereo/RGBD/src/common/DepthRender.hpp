#ifndef PERCIPIO_SAMPLE_COMMON_DEPTH_RENDER_HPP_
#define PERCIPIO_SAMPLE_COMMON_DEPTH_RENDER_HPP_

#include <opencv2/opencv.hpp>
#include <map>
#include <vector>

class DepthRender {
public:
    enum OutputColorType {
        COLORTYPE_RAINBOW = 0,
        COLORTYPE_BLUERED = 1,
        COLORTYPE_GRAY = 2
    };

    enum ColorRangeMode {
        COLOR_RANGE_ABS = 0,
        COLOR_RANGE_DYNAMIC = 1
    };

    DepthRender() : needResetColorTable(true)
                  , color_type(COLORTYPE_BLUERED)
                  , range_mode(COLOR_RANGE_DYNAMIC)
                  , min_distance(0)
                  , max_distance(0)
                  , invalid_label(0)
                  {}

    void SetColorType( OutputColorType ct = COLORTYPE_BLUERED ){
                if(ct != color_type){
                    needResetColorTable = true;
                    color_type = ct;
                }
            }

    void SetRangeMode( ColorRangeMode rm = COLOR_RANGE_DYNAMIC ){
                if(range_mode != rm){
                    needResetColorTable = true;
                    range_mode = rm;
                }
            }

    /// for abs mode
    void SetColorRange(int minDis, int maxDis){
                min_distance = minDis;
                max_distance = maxDis;
            }

    /// input 16UC1 output 8UC3
    void    Compute(const cv::Mat &src, cv::Mat& dst ){
                dst = Compute(src);
            }
    cv::Mat Compute(const cv::Mat &src){
                cv::Mat src16U;
                if(src.type() != CV_16U){
                    src.convertTo(src16U, CV_16U);
                }else{
                    src16U = src;
                }

                if(needResetColorTable){
                    BuildColorTable();
                    needResetColorTable = false;
                }

                cv::Mat dst;
                filtered_mask = (src16U == invalid_label);
                clr_disp = src16U.clone();
                if(COLOR_RANGE_ABS == range_mode) {
                    TruncValue(clr_disp, filtered_mask, min_distance, max_distance);
                    clr_disp -= min_distance;
                    clr_disp = clr_disp * 255 / (max_distance - min_distance);
                    clr_disp.convertTo(clr_disp, CV_8UC1);
                } else {
                    short vmax, vmin;
                    HistAdjustRange(clr_disp, invalid_label, min_distance, vmin, vmax);
                    clr_disp = (clr_disp - vmin) * 255 / (vmax - vmin);
                    //clr_disp = 255 - clr_disp;
                    clr_disp.convertTo(clr_disp, CV_8UC1);
                }

                switch (color_type) {
                case COLORTYPE_GRAY:
                    clr_disp = 255 - clr_disp;
                    cv::cvtColor(clr_disp, dst, CV_GRAY2BGR);
                    break;
                case COLORTYPE_BLUERED:
                    //temp = 255 - clr_disp;
                    CalcColorMap(clr_disp, dst);
                    //cv::applyColorMap(temp, color_img, cv::COLORMAP_COOL);
                    break;
                case COLORTYPE_RAINBOW:
                    //cv::cvtColor(color_img, color_img, CV_GRAY2BGR);
                    cv::applyColorMap(clr_disp, dst, cv::COLORMAP_RAINBOW);
                    break;
                }
                ClearInvalidArea(dst, filtered_mask);

                return dst;
            }

private:
    void CalcColorMap(const cv::Mat &src, cv::Mat &dst){
                std::vector<cv::Scalar> &table = _color_lookup_table;
                assert(table.size() == 256);
                assert(!src.empty());
                assert(src.type() == CV_8UC1);
                dst.create(src.size(), CV_8UC3);
                const unsigned char* sptr = src.ptr<unsigned char>();
                unsigned char* dptr = dst.ptr<unsigned char>();
                for (int i = src.size().area(); i != 0; i--) {
                    cv::Scalar &v = table[*sptr];
                    dptr[0] = (unsigned char)v.val[0];
                    dptr[1] = (unsigned char)v.val[1];
                    dptr[2] = (unsigned char)v.val[2];
                    dptr += 3;
                    sptr += 1;
                }
            }
    void BuildColorTable(){
                _color_lookup_table.resize(256);
                cv::Scalar from(50, 0, 0xff), to(50, 200, 255);
                for (int i = 0; i < 128; i++) {
                    float a = (float)i / 128;
                    cv::Scalar &v = _color_lookup_table[i];
                    for (int j = 0; j < 3; j++) {
                        v.val[j] = from.val[j] * (1 - a) + to.val[j] * a;
                    }
                }
                from = to;
                to = cv::Scalar(255, 104, 0);
                for (int i = 128; i < 256; i++) {
                    float a = (float)(i - 128) / 128;
                    cv::Scalar &v = _color_lookup_table[i];
                    for (int j = 0; j < 3; j++) {
                        v.val[j] = from.val[j] * (1 - a) + to.val[j] * a;
                    }
                }
            }
    //keep value in range
    void TruncValue(cv::Mat &img, cv::Mat &mask, short min_val, short max_val){
                assert(max_val >= min_val);
                assert(img.type() == CV_16SC1);
                assert(mask.type() == CV_8UC1);
                short* ptr = img.ptr<short>();
                unsigned char* mask_ptr = mask.ptr<unsigned char>();
                for (int i = img.size().area(); i != 0; i--) {
                  if (*ptr > max_val) {
                    *ptr = max_val;
                    *mask_ptr = 0xff;
                  } else if (*ptr < min_val) {
                    *ptr = min_val;
                    *mask_ptr = 0xff;
                  }
                  ptr++;
                  mask_ptr++;
                }
            }
    void ClearInvalidArea(cv::Mat &clr_disp, cv::Mat &filtered_mask){
                assert(clr_disp.type() == CV_8UC3);
                assert(filtered_mask.type() == CV_8UC1);
                assert(clr_disp.size().area() == filtered_mask.size().area());
                unsigned char* filter_ptr = filtered_mask.ptr<unsigned char>();
                unsigned char* ptr = clr_disp.ptr<unsigned char>();
                int len = clr_disp.size().area();
                for (int i = 0; i < len; i++) {
                    if (*filter_ptr != 0) {
                      ptr[0] = ptr[1] = ptr[2] = 0;
                    }
                    filter_ptr++;
                    ptr += 3;
                }
            }
    void HistAdjustRange(const cv::Mat &dist, short invalid, int min_display_distance_range
            , short &min_val, short &max_val) {
                std::map<short, int> hist;
                int sz = dist.size().area();
                const short* ptr = dist.ptr < short>();
                int total_num = 0;
                for (int idx = sz; idx != 0; idx--, ptr++) {
                    if (invalid == *ptr) {
                        continue;
                    }
                    total_num++;
                    if (hist.find(*ptr) != hist.end()) {
                        hist[*ptr]++;
                    } else {
                        hist.insert(std::make_pair(*ptr, 1));
                    }
                }
                if (hist.empty()) {
                    min_val = 0;
                    max_val = 2000;
                    return;
                }
                const int delta = total_num * 0.01;
                int sum = 0;
                min_val = hist.begin()->first;
                for (std::map<short, int>::iterator it = hist.begin(); it != hist.end();it++){
                    sum += it->second;
                    if (sum > delta) {
                        min_val = it->first;
                        break;
                    }
                }

                sum = 0;
                max_val = hist.rbegin()->first;
                for (std::map<short, int>::reverse_iterator s = hist.rbegin()
                        ; s != hist.rend(); s++) {
                    sum += s->second;
                    if (sum > delta) {
                        max_val = s->first;
                        break;
                    }
                }

                const int min_display_dist = min_display_distance_range;
                if (max_val - min_val < min_display_dist) {
                    int m = (max_val + min_val) / 2;
                    max_val = m + min_display_dist / 2;
                    min_val = m - min_display_dist / 2;
                    if (min_val < 0) {
                        min_val = 0;
                    }
                }
            }

    bool            needResetColorTable;
    OutputColorType color_type;
    ColorRangeMode  range_mode;
    int             min_distance;
    int             max_distance;
    uint16_t        invalid_label;
    cv::Mat         clr_disp ;
    cv::Mat         filtered_mask;
    std::vector<cv::Scalar> _color_lookup_table;
};

#endif
