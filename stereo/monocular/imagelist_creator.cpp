/*this creates a yaml or xml list of files from the command line args
 * 由图片文件夹生成 图片文件路径 yaml文件
 * 示例用法 ./imagelist_creator wimagelist.yaml ../t/*JPG
 */

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <iostream>

using std::string;
using std::cout;
using std::endl;

using namespace cv;

static void help(char** av)//argv参数
{
  cout << "\nThis creates a yaml or xml list of files from the command line args\n"
      "usage:\n./" << av[0] << " imagelist.yaml *.png\n"  //命令行  用法
      << "Try using different extensions.(e.g. yaml yml xml xml.gz etc...)\n"
      << "This will serialize this list of images or whatever with opencv's FileStorage framework" << endl;
}

int main(int ac, char** av)
{
  cv::CommandLineParser parser(ac, av, "{help h||}{@output||}");//解析参数  得到帮助参数 和 输出文件名
  if (parser.has("help"))//有帮助参数
  {
    help(av);//显示帮助信息
    return 0;
  }
  string outputname = parser.get<string>("@output");

  if (outputname.empty())
  {
    help(av);
    return 1;
  }

  Mat m = imread(outputname); //check if the output is an image - prevent overwrites!
  if(!m.empty()){
    std::cerr << "fail! Please specify an output file, don't want to overwrite you images!" << endl;
    help(av);
    return 1;
  }

  FileStorage fs(outputname, FileStorage::WRITE);//opencv 读取文件类
  fs << "images" << "[";//文件流重定向
  for(int i = 2; i < ac; i++){
    fs << string(av[i]);//从第二个参数开始为 图片名
  }
  fs << "]";
  return 0;
}
