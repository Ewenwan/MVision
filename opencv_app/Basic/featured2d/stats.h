/*
状态类 操作
*/
#ifndef STATS_H
#define STATS_H

struct Stats
{
    int matches;//初始匹配点对数量
    int inliers;//内点数量 较好的匹配点数量
    double ratio;//内点所占比率
    int keypoints;//关键点数量
    // 状态类初始化
    Stats() : matches(0),
        inliers(0),
        ratio(0),
        keypoints(0)
    {}
    // 运算符重定义  +=
    Stats& operator+=(const Stats& op) {
        matches += op.matches;//家和
        inliers += op.inliers;
        ratio += op.ratio;
        keypoints += op.keypoints;
        return *this;
    }
    // 运算符重定义  /=  除以所给 数num
    Stats& operator/=(int num)
    {
        matches /= num;
        inliers /= num;
        ratio /= num;
        keypoints /= num;
        return *this;
    }
};

#endif // STATS_H
