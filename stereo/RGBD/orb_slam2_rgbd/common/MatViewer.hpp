#ifndef XYZ_MAT_VIEWER_HPP_
#define XYZ_MAT_VIEWER_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include "DepthRender.hpp"


class OpencvViewer
{
public:
    static void drawText(cv::Mat& img, const std::string& text, const cv::Point loc,
                         double scale, const cv::Scalar& color, int thickness);

    virtual void show(const std::string& win, const cv::Mat& img);
    virtual void onMouseCallback(cv::Mat& img, int event, const cv::Point pnt);

private:
    static void __onMouseCallback(int event, int x, int y, int flags, void* ustc);

    cv::Mat _img;
    std::string _win;
    void*   _ustc;
};

inline void OpencvViewer::drawText(cv::Mat& img, const std::string& text
        , const cv::Point loc, double scale, const cv::Scalar& color, int thickness)
{
    cv::putText(img, text, loc, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness);
}



class GraphicItem
{
public:
    GraphicItem() : _id(++globalID), _color(255,255,255) {}
    virtual ~GraphicItem() {}

    void setColor(const cv::Scalar& color) { _color = color; }

    int id() const { return _id; }
    cv::Scalar color() const { return _color; }
    virtual void draw(cv::Mat& img) = 0;

private:
    static int globalID;

    int         _id;
    cv::Scalar  _color;
};

class GraphicRectangleItem : public GraphicItem
{
public:
    cv::Rect _rect;

    GraphicRectangleItem() : GraphicItem(), _rect() {}
    GraphicRectangleItem(const cv::Rect& rect) : GraphicItem(), _rect(rect) {}
    virtual ~GraphicRectangleItem() {}
    virtual void draw(cv::Mat& img){ cv::rectangle(img, _rect, color()); }
};


class DepthViewer : public OpencvViewer
{
public:
    static std::string depthStringAtLoc(const cv::Mat& img, const cv::Point pnt);

    void addGraphicItem(GraphicItem* item) {
                    _items.insert(std::make_pair(item->id(), item));}
    void delGraphicItem(GraphicItem* item) { _items.erase(item->id()); }

    virtual void show(const std::string& win, const cv::Mat& depthImage);
    virtual void onMouseCallback(cv::Mat& img, int event, const cv::Point pnt);

private:
    void drawFixLoc(cv::Mat& img);
    void drawItems(cv::Mat& img);

    DepthRender _render;
    cv::Mat _img;
    cv::Point   _fixLoc;
    std::map<int, GraphicItem*> _items;
};


#endif
