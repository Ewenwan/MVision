/*
最小分割算法  (分割点云)
该算法是将一幅点云图像分割为两部分：
前景点云（目标物体）和背景物体（剩余部分）
[参考](http://www.cnblogs.com/li-yao7758258/p/6696953.html)
[论文的地址](http://gfx.cs.princeton.edu/pubs/Golovinskiy_2009_MBS/paper_small.pdf)
The Min-Cut (minimum cut) algorithm最小割算法是图论中的一个概念，
其作用是以某种方式，将两个点分开，当然这两个点中间可能是通过无数的点再相连的。

如果要分开最左边的点和最右边的点，红绿两种割法都是可行的，
但是红线跨过了三条线，绿线只跨过了两条。
单从跨线数量上来论可以得出绿线这种切割方法更优 的结论。
但假设线上有不同的权值，那么最优切割则和权值有关了。
当你给出了点之间的 “图” ，以及连线的权值时，
最小割算法就能按照要求把图分开。

所以那么怎么来理解点云的图呢？
显而易见，切割有两个非常重要的因素，
第一个是获得点与点之间的拓扑关系，这种拓扑关系就是生成一张 “图”。
第二个是给图中的连线赋予合适的权值。
只要这两个要素合适，最小割算法就会正确的分割出想要的结果。
点云是分开的点。只要把点云中所有的点连起来就可以了。

	连接算法如下：
	   1. 找到每个点临近的n个点
	   2. 将这n个点和父点连接
	   3. 找到距离最小的两个块（A块中某点与B块中某点距离最小），并连接
	   4. 重复3，直至只剩一个块
经过上面的步骤现在已经有了点云的“图”，只要给图附上合适的权值，就满足了最小分割的前提条件。
物体分割比如图像分割给人一个直观印象就是属于该物体的点，应该相互之间不会太远。
也就是说，可以用点与点之间的欧式距离来构造权值。
所有线的权值可映射为线长的函数。 

cost = exp(-(dist/cet)^2)  距离越远　cost越小　越容易被分割

我们知道这种分割是需要指定对象的，也就是我们指定聚类的中心点（center）以及聚类的半径（radius），
当然我们指定了中心点和聚类的半径，那么就要被保护起来，保护的方法就是增加它的权值.

dist2Center / radius

dist2Center　＝　sqrt((x-x_center)^2+(y-y_center)^2)



*/
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <iostream>
#include <pcl/segmentation/region_growing_rgb.h>

int
main(int argc, char** argv)
{
    //申明点云的类型
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // 法线
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../min_Cut_Based.pcd", *cloud) != 0)
    {
        return -1;
    }
       // 申明一个Min-cut的聚类对象
    pcl::MinCutSegmentation<pcl::PointXYZ> clustering;
    clustering.setInputCloud(cloud);//设置输入
    //创建一个点云，列出所知道的所有属于对象的点 
    // （前景点）在这里设置聚类对象的中心点（想想是不是可以可以使用鼠标直接选择聚类中心点的方法呢？）
    pcl::PointCloud<pcl::PointXYZ>::Ptr foregroundPoints(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointXYZ point;
    point.x = 100.0;
    point.y = 100.0;
    point.z = 100.0;
    foregroundPoints->points.push_back(point);
    clustering.setForegroundPoints(foregroundPoints);//设置聚类对象的前景点
       
    //设置sigma，它影响计算平滑度的成本。它的设置取决于点云之间的间隔（分辨率）
    clustering.setSigma(0.02);// cet cost = exp(-(dist/cet)^2) 
    // 设置聚类对象的半径.
    clustering.setRadius(0.01);// dist2Center / radius

         //设置需要搜索的临近点的个数，增加这个也就是要增加边界处图的个数
    clustering.setNumberOfNeighbours(20);

        //设置前景点的权重（也就是排除在聚类对象中的点，它是点云之间线的权重，）
    clustering.setSourceWeight(0.6);

    std::vector <pcl::PointIndices> clusters;
    clustering.extract(clusters);

    std::cout << "Maximum flow is " << clustering.getMaxFlow() << "." << std::endl;

    int currentClusterNum = 1;
    for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
    {
        //设置聚类后点云的属性
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
            cluster->points.push_back(cloud->points[*point]);
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;

           //保存聚类的结果
        if (cluster->points.size() <= 0)
            break;
        std::cout << "Cluster " << currentClusterNum << " has " << cluster->points.size() << " points." << std::endl;
        std::string fileName = "cluster" + boost::to_string(currentClusterNum) + ".pcd";
        pcl::io::savePCDFileASCII(fileName, *cluster);

        currentClusterNum++;
    }

}
