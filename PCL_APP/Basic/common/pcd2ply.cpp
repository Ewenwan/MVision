/*
PCL中的常用的点云存储文件为.pcd文件，
但是很多场合下使用的点云存储文件为.ply文件，
特别是要在meshlab中查看的时候。

PCL中有相应的类PLYWriter可以帮助我们实现从.pcd文件到.ply文件的转换。
值得一提的是，在PCL的源码pcl-master中的tools文件夹里有很多范例的例程，
可以帮助我们参照着实现很多简单但基本的功能。
其中，pcd2ply.cpp便是指导我们实现从.pcd文件到.ply文件的转换的一个cpp文件。

以下的小程序便是根据上述的指导文件写成的。值得注意的是，
以下将用到的 PCLPointCloud2 类并不适用于PCL官网上的pcl-1.6.0版本，
因为pcl-1.6.0实在是太老了，并没有实现这个类。
解决方法是使用较新的pcl-1.7.2版本，这样就可以了。


PDC文件格式
    点云数据可以用ASCII码的形式存储在PCD文件中（
    关于该格式的描述可以参考链接：The PCD (Point Cloud Data) file format）。
    为了生成三维点云数据，在excel中用rand()函数生成200行0-1的小数，ABC三列分别代表空间点的xyz坐标。

    PDC文件格式
    # .PCD v.7 - Point Cloud Data file format
    VERSION .7        
    FIELDS x y z     
    SIZE 4 4 4         
    TYPE F F F         
    COUNT 1 1 1     
    WIDTH 200        
    HEIGHT 1        
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS 200        
    DATA ascii        
    0.88071666    0.369209703    0.062937221
    0.06418104    0.579762553    0.221359779
    ...
    ...
    0.640053058    0.480279041    0.843647334
    0.245554712    0.825770496    0.626442137

    内容格式
    FIELDS x y z                                # XYZ data
    FIELDS x y z rgb                            # XYZ + colors
    FIELDS x y z normal_x normal_y normal_z     # XYZ + surface normals
    FIELDS j1 j2 j3                             # moment invariants
    存储大小
    SIZE - specifies the size of each dimension in bytes. Examples:

        unsigned char/char has 1 byte
        unsigned short/short has 2 bytes
        unsigned int/int/float has 4 bytes
        double has 8 bytes
    数据类型
    TYPE - specifies the type of each dimension as a char. The current accepted types are:

        I - represents signed types int8 (char), int16 (short), and int32 (int)
        U - represents unsigned types uint8 (unsigned char), uint16 (unsigned short), uint32 (unsigned int)
        F - represents float types

    视角
    The viewpoint information is specified as a translation (tx ty tz) + quaternion (qw qx qy qz). The default value is:
    VIEWPOINT 0 0 0 1 0 0 0


CSV格式的ply文件类
PLY是一种电脑档案格式，全名为多边形档案（Polygon File Format）或 
斯坦福三角形档案（Stanford Triangle Format）。 

该格式主要用以储存立体扫描结果的三维数值，透过多边形片面的集合描述三维物体，
与其他格式相较之下这是较为简单的方法。它可以储存的资讯包含颜色、
透明度、表面法向量、材质座标与资料可信度，并能对多边形的正反两面设定不同的属性。

在档案内容的储存上PLY有两种版本，分别是纯文字（ASCII）版本与二元码（binary）版本，
其差异在储存时是否以ASCII编码表示元素资讯。

每个PLY档都包含档头（header），用以设定网格模型的“元素”与“属性”，
以及在档头下方接着一连串的元素“数值资料”。
一般而言，网格模型的“元素”
就是顶点（vertices）、
面（faces），另外还可能包含有
边（edges）、
深度图样本（samples of range maps）与
三角带（triangle strips）等元素。
无论是纯文字与二元码的PLY档，档头资讯都是以ASCII编码编写，
接续其后的数值资料才有编码之分。PLY档案以此行：

ply     开头作为PLY格式的识别。接着第二行是版本资讯，目前有三种写法：

format ascii 1.0
format binary_little_endian 1.0
format binary_big_endian 1.0
       // 其中ascii, binary_little_endian, binary_big_endian是档案储存的编码方式，
       // 而1.0是遵循的标准版本（现阶段仅有PLY 1.0版）。

comment This is a comment!   // 使用'comment'作为一行的开头以编写注解
comment made by anonymous
comment this file is a cube

element vertex 8    //  描述元素  element <element name> <number in file>   8个顶点
                    //  以下 接续的6行property描述构成vertex元素的数值字段顺序代表的意义，及其资料形态。
property float32 x  //  描述属性  property <data_type> <property name 1>
property float32 y  //  每个顶点使用3个 float32 类型浮点数（x，y，z）代表点的坐标
property float32 z

property uchar blue // 使用3个unsigned char代表顶点颜色，颜色顺序为 (B, G, R)
property uchar green
property uchar red

element face 12       
property list uint8 int32 vertex_index
 			// 12 个面(6*2)   另一个常使用的元素是面。
			// 由于一个面是由3个以上的顶点所组成，因此使用一个“顶点列表”即可描述一个面, 
			// PLY格式使用一个特殊关键字'property list'定义之。 
end_header              // 最后，标头必须以此行结尾：

// 档头后接着的是元素资料（端点座标、拓朴连结等）。在ASCII格式中各个端点与面的资讯都是以独立的一行描述

0 0 0                   // 8个顶点 索引 0~7
0 25.8 0
18.9 0 0
18.9 25.8 0
0 0 7.5
0 25.8 7.5
18.9 0 7.5
18.9 25.8 7.5

3 5 1 0            // 前面的3表示3点表示的面   有的一个面 它用其中的三个点 表示了两次  6*2=12
3 5 4 0            // 后面是上面定点的 索引 0~7
3 4 0 2
3 4 6 2
3 7 5 4
3 7 6 4
3 3 2 1
3 1 2 0
3 5 7 1
3 7 1 3
3 7 6 3
3 6 3 2


*/
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

void
printHelp (int, char **argv)
{
  print_error ("Syntax is: %s [-format 0|1] [-use_camera 0|1] input.pcd output.ply\n", argv[0]);
}

bool
loadCloud (const std::string &filename, pcl::PCLPointCloud2 &cloud)
{
  TicToc tt;
  print_highlight ("Loading "); print_value ("%s ", filename.c_str ());

  tt.tic ();
  if (loadPCDFile (filename, cloud) < 0)
    return (false);
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
  print_info ("Available dimensions: "); print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());

  return (true);
}

void
saveCloud (const std::string &filename, const pcl::PCLPointCloud2 &cloud, bool binary, bool use_camera)
{
  TicToc tt;
  tt.tic ();

  print_highlight ("Saving "); print_value ("%s ", filename.c_str ());
  
  pcl::PLYWriter writer;
  writer.write (filename, cloud, Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), binary, use_camera);
  
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
}

/* ---[ */
int
main (int argc, char** argv)
{
  print_info ("Convert a PCD file to PLY format. For more information, use: %s -h\n", argv[0]);

  if (argc < 3)
  {
    printHelp (argc, argv);
    return (-1);
  }

  // Parse the command line arguments for .pcd and .ply files
  std::vector<int> pcd_file_indices = parse_file_extension_argument (argc, argv, ".pcd");
  std::vector<int> ply_file_indices = parse_file_extension_argument (argc, argv, ".ply");
  if (pcd_file_indices.size () != 1 || ply_file_indices.size () != 1)
  {
    print_error ("Need one input PCD file and one output PLY file.\n");
    return (-1);
  }

  // Command line parsing
  bool format = true;
  bool use_camera = true;
  parse_argument (argc, argv, "-format", format);
  parse_argument (argc, argv, "-use_camera", use_camera);
  print_info ("PLY output format: "); print_value ("%s, ", (format ? "binary" : "ascii"));
  print_value ("%s\n", (use_camera ? "using camera" : "no camera"));

  // Load the first file
  pcl::PCLPointCloud2 cloud;
  if (!loadCloud (argv[pcd_file_indices[0]], cloud)) 
    return (-1);

  // Convert to PLY and save
  saveCloud (argv[ply_file_indices[0]], cloud, format, use_camera);

  return (0);
}
