/*
PCA（Principal Component Analysis）主成分分析
主要用于数据降维

样本  协方差矩阵 的特征值和特征向量  取前四个特征值所对应的特征向量
特征矩阵 投影矩阵


对于一组样本的feature组成的多维向量，多维向量里的某些元素本身没有区分
我们的目的是找那些变化大的元素，即方差大的那些维，
而去除掉那些变化不大的维，从而使feature留下的都是最能代表此元素的“精品”，
而且计算量也变小了。

对于一个k维的feature来说，相当于它的每一维feature与
其他维都是正交的（相当于在多维坐标系中，坐标轴都是垂直的），
那么我们可以变化这些维的坐标系，从而使这个feature在某些维上方差大，
而在某些维上方差很


求得一个k维特征的投影矩阵，这个投影矩阵可以将feature从高维降到低维。
投影矩阵也可以叫做变换矩阵。新的低维特征必须每个维都正交，特征向量都是正交的

通过求样本矩阵的协方差矩阵，然后求出协方差矩阵的特征向量，这些特征向量就可以构成这个投影矩阵了。
特征向量的选择取决于协方差矩阵的特征值的大

对于一个训练集，100个样本，feature是10维，那么它可以建立一个100*10的矩阵，作为样本。
求这个样本的协方差矩阵，得到一个10*10的协方差矩阵，然后求出这个协方差矩阵的特征值和特征向量，
应该有10个特征值和特征向量，我们根据特征值的大小，取前四个特征值所对应的特征向量，
构成一个10*4的矩阵，这个矩阵就是我们要求的特征矩阵，100*10的样本矩阵乘以这个10*4的特征矩阵，
就得到了一个100*4的新的降维之后的样本矩阵，每个样本的维数下



*/
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

// Function declarations
void drawAxis(Mat&, Point, Point, Scalar, const float);
double getOrientation(const vector<Point> &, Mat&);

/**
 * @function drawAxis
 */
void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
//! [visualization1]
    double angle;
    double hypotenuse;
    angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
//    double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
//    cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range

    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, colour, 1, LINE_AA);

    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);

    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);
//! [visualization1]
}

/**
 * @function getOrientation
 */
double getOrientation(const vector<Point> &pts, Mat &img)
{
//! [pca]
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }

    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);

    //Store the center of the object
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                      static_cast<int>(pca_analysis.mean.at<double>(0, 1)));

    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; ++i)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));

        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    }

//! [pca]
//! [visualization]
    // Draw the principal components
    circle(img, cntr, 3, Scalar(255, 0, 255), 2);
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    drawAxis(img, cntr, p1, Scalar(0, 255, 0), 1);
    drawAxis(img, cntr, p2, Scalar(255, 255, 0), 5);

    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
//! [visualization]

    return angle;
}

/**
 * @function main
 */
int main(int argc, char** argv)
{
//! [pre-process]
    // Load image
    CommandLineParser parser(argc, argv, "{@input | ../../common/data/pca_test1.jpg | input image}");
    parser.about( "This program demonstrates how to use OpenCV PCA to extract the orienation of an object.\n" );
    parser.printMessage();

    Mat src = imread(parser.get<String>("@input"));

    // Check if image is loaded successfully
    if(src.empty())
    {
        cout << "Problem loading image!!!" << endl;
        return EXIT_FAILURE;
    }

    imshow("src", src);

    // Convert image to grayscale
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Convert image to binary
    Mat bw;
    threshold(gray, bw, 50, 255, THRESH_BINARY | THRESH_OTSU);
//! [pre-process]

//! [contours]
    // Find all the contours in the thresholded image
    vector<vector<Point> > contours;
    findContours(bw, contours, RETR_LIST, CHAIN_APPROX_NONE);

    for (size_t i = 0; i < contours.size(); ++i)
    {
        // Calculate the area of each contour
        double area = contourArea(contours[i]);
        // Ignore contours that are too small or too large
        if (area < 1e2 || 1e5 < area) continue;

        // Draw each contour only for visualisation purposes
        drawContours(src, contours, static_cast<int>(i), Scalar(0, 0, 255), 2, LINE_8);
        // Find the orientation of each shape
        getOrientation(contours[i], src);
    }
//! [contours]

    imshow("output", src);

    waitKey(0);
    return 0;
}
