//李飞，科大先研院
//sume.cn@aliyun.com
//just for fun
// 万有文，修改
#include <stdio.h>
#include <vector>
#include <time.h>
using namespace std;
#include <opencv2/opencv.hpp>
using namespace cv;


#define NUM_PARTICLES 50 // 点数量

// set laser length as 80
// map resolution is 0.05m, so 80 pixel is 4m
// 根据地图，给出所给坐标处的，雷达扫描
#define LASER_RADIUS 80
const vector<int> compute_scan(int x, int y, const Mat & map)
{
    vector<int> scan;

    for (int theta = 0; theta < 360; ++theta)// 360度
    {
        int r = 0;
        for (; r < LASER_RADIUS; ++r)// 扫描半径
        {
            int dx = r*cos(theta*2*CV_PI/360.0);
            int dy = r*sin(theta*2*CV_PI/360.0);

            if (map.at<unsigned char>(y+dy, x+dx) != 255) // 255 黑色为 有障碍物
            {
                break;// 跳过无障碍物
            }
        }

        scan.push_back(r);// 记录有障碍物处的 半径（360度上，每一度上障碍物处的半径）
    }

    return scan;
}

// 两数组相似度==================================================
double compute_score(const vector<int> v1, const vector<int> v2)
{
    // CV_ASSERT(v1.size() == v2.size());

    double score = 0.0;
    for (int i = 0; i < v1.size(); ++i)
    {
        score += pow(v1[i]-v2[i], 2.0);// 差平方和
    }

    score = sqrt(score);// 开平方

    return score;
}


int rand_pn(int r)
{
    if (rand()%2 == 0)
    {
        return rand()%r;
    }
    else
    {
        return -rand()%r;
    }
}


int main()
{
    srand((unsigned int)time(NULL));// 随机数种子

    Mat map = imread("map.bmp", CV_LOAD_IMAGE_GRAYSCALE);// 地图 372*361

    vector<Point> particles;// 二维坐标点
    for (int i = 0; i < NUM_PARTICLES; ++i)
    {
        int xx = 0;
        int yy = 0;
        do
        {
            xx = rand()%map.cols;// 无障碍物区域 随机产生点坐标
            yy = rand()%map.rows;
        } 
        while (map.at<unsigned char>(yy,xx) != 255); // 无障碍物区域===

        particles.push_back(Point(xx,yy));// 产生的随机粒子
    }

    ///////////////////////////////////////////////////////////////

    Point point_start(30,330);// 起点
    Point point_end(240,330); // 终点

    double score[NUM_PARTICLES] = {0};// 每个粒子对应一个得分

    for (int x = 30; x < 240; ++x)
    {
        // real position: x, 330  纵坐标y == 330
        // 规划的直线路径 point_start ----> point_end
        vector<int> real_scan = compute_scan(x, 330, map);// 真实轨迹点 周围 360度 障碍物处的 半径 测量信息

        int offset = x-30;

        // compute each particle
        for (int i = 0; i < NUM_PARTICLES; ++i)
        {
            vector<int> v = compute_scan(particles[i].x+offset, particles[i].y, map);// 随机点处的 测量信息
            score[i] = compute_score(real_scan, v);// 计算 随机点 和 真实点 雷达测量信息的 匹配度
        }

        // kill worst particles and replace with mutations of the best
        double *maxLoc = max_element(score, score+NUM_PARTICLES);  
        double *minLoc = min_element(score, score+NUM_PARTICLES); // 得分越小，匹配越好
        printf("max: %f, min: %f\n", *maxLoc, *minLoc);

        {
            int xx = 0;
            int yy = 0;
            do
            {
                xx = particles[minLoc-score].x + rand_pn(20);// 用好的点 产生新的点
                yy = particles[minLoc-score].y + rand_pn(20);
            } while(map.at<unsigned char>(yy,xx) != 255);

            particles[maxLoc-score].x = xx;// 差的点被替换掉
            particles[maxLoc-score].y = yy;
        }
        
        // 保存每一步的地图=========
        {
            Mat mat_disp = map.clone();
            for (int i = 0; i < NUM_PARTICLES; ++i)
            {
                circle(mat_disp, Point(particles[i].x+offset, particles[i].y), 2, Scalar(0));// 画点
            }
            char str[30];
            sprintf(str, "disp%d.png", x);// 产生字符串，图片名称
            imwrite(str, mat_disp);       // 保存地图
        }
    }
}

