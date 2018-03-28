/*
文件读写
参数文件设置

自定义数据类  MyData
包含一个int 变量 一个 double变量 一个string变量
提供读写文件的方法


字符串序列 string ["","",""]
字典映射数据 mapping{ }

%YAML:1.0
iterationNr: 100
strings:
   - "image1.jpg"
   - Awesomeness
   - "../data/baboon.jpg"
Mapping:
   One: 1
   Two: 2
R: !!opencv-matrix
   rows: 3
   cols: 3
   dt: u
   data: [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ]
T: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ 0., 0., 0. ]
MyData:
   A: 97
   X: 3.1415926535897931e+00
   id: mydata1234

*/
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

static void help(char** av)
{
    cout << endl
        << av[0] << " shows the usage of the OpenCV serialization functionality."         << endl
        << "usage: "                                                                      << endl
        <<  av[0] << " outputfile.yml.gz"                                                 << endl
        << "The output file may be either XML (xml) or YAML (yml/yaml). You can even compress it by "
        << "specifying this in its extension like xml.gz yaml.gz etc... "                  << endl
        << "With FileStorage you can serialize objects in OpenCV by using the << and >> operators" << endl
        << "For example: - create a class and have it serialized"                         << endl
        << "             - use it to read and write matrices."                            << endl;
}

//自定义数据类  MyData
class MyData
{
public:
    MyData() : A(0), X(0), id()
    {}
    explicit MyData(int) : A(97), X(CV_PI), id("mydata1234") // explicit 显式避免隐式转换  只能用在类构造函数
// 比如类A 只有一个int变量   则 可进行 A tep_a = 10;//定义方式 这里包含了将 整形10 隐式的转换成 类A类型

//写文件方法
    {}
    void write(FileStorage& fs) const                        //Write serialization for this class
    {
        fs << "{" << "A" << A << "X" << X << "id" << id << "}";
    }
// 读文件方法
    void read(const FileNode& node)                          //Read serialization for this class
    {
        A = (int)node["A"];
        X = (double)node["X"];
        id = (string)node["id"];
    }
public:   // 数据成员 Data Members
    int A;
    double X;
    string id;
};

//These write and read functions must be defined for the serialization in FileStorage to work


static void write(FileStorage& fs, const std::string&, const MyData& x)
{
    x.write(fs);
}

static void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()){
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

// This function will print our custom class to the console
// 输出流 操作符 重定义
static ostream& operator<<(ostream& out, const MyData& m)
{
    out << "{ id = " << m.id << ", ";
    out << "X = " << m.X << ", ";
    out << "A = " << m.A << "}";
    return out;
}

//主函数
int main(int argc, char** argv)
{
    if (argc != 2)
        help(argv);
//写文件
   // const char* filename = argc >=2 ? argv[1] : "../data/lena.jpg";
   // string filename = av[1];//文件吗
    string filename = argc >=2 ? argv[1] : "test.yaml"; 
    { //write
        Mat R = Mat_<uchar>::eye(3, 3),
            T = Mat_<double>::zeros(3, 1);
        MyData m(1);

        FileStorage fs(filename, FileStorage::WRITE);

        fs << "iterationNr" << 100;
        fs << "strings" << "[";      // text - 字符串序列 string sequence [
        fs << "image1.jpg" << "Awesomeness" << "../data/baboon.jpg";
        fs << "]";                   // close sequence }

        fs << "Mapping";             // text - 映射结构 字典结构 mapping
        fs << "{" << "One" << 1;
        fs <<        "Two" << 2 << "}";

        fs << "R" << R;             // cv::Mat
        fs << "T" << T;

        fs << "MyData" << m;        // 自己的数据结构   输出流 操作符 重定义

        fs.release();               // 显示关闭 文件 explicit close
        cout << "Write Done." << endl;
    }
//读文件
    {//read
        cout << endl << "Reading: " << endl;
        FileStorage fs;
        fs.open(filename, FileStorage::READ);//打开文件 读取

        int itNr;
        //fs["iterationNr"] >> itNr;
        itNr = (int) fs["iterationNr"];
        cout << itNr;
        if (!fs.isOpened())
        {
            cerr << "Failed to open " << filename << endl;
            help(argv);
            return 1;
        }

        FileNode n = fs["strings"];    //字符串序列  Read string sequence - Get node
        if (n.type() != FileNode::SEQ)
        {
            cerr << "strings is not a sequence! FAIL" << endl;
            return 1;
        }

        FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
        for (; it != it_end; ++it)
            cout << (string)*it << endl;//打印字符串序列中的 每一个字符串


        n = fs["Mapping"];                                // 读取映射 值
        cout << "Two  " << (int)(n["Two"]) << "; ";
        cout << "One  " << (int)(n["One"]) << endl << endl;


        MyData m;
        Mat R, T;

        fs["R"] >> R;                                     // 读取cv::Mat 数据 Read cv::Mat
        fs["T"] >> T;
        fs["MyData"] >> m;                                // 读取自己的数据结构 Read your own structure_

        cout << endl
            << "R = " << R << endl;
        cout << "T = " << T << endl << endl;
        cout << "MyData = " << endl << m << endl << endl;// 自己的数据结构   输出流 操作符 重定义

        //Show default behavior for non existing nodes
        cout << "Attempt to read NonExisting (should initialize the data structure with its default).";
        fs["NonExisting"] >> m;
        cout << endl << "NonExisting = " << endl << m << endl;
    }

    cout << endl
        << "Tip: Open up " << filename << " with a text editor to see the serialized data." << endl;

    return 0;
}
