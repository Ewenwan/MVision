/*
 *myslam::Config 配置文件
 */

#include "myslam/config.h"

namespace myslam //命令空间下 防止定义的出其他库里的同名函数
{
    //   读取参数文件
    void Config::setParameterFile( const std::string& filename )
    {
	if ( config_ == nullptr )//为空
	    config_ = shared_ptr<Config>(new Config);//初始化一个
	config_->file_ = cv::FileStorage( filename.c_str(), cv::FileStorage::READ );//读文件
	if ( config_->file_.isOpened() == false )//打开文件错误
	{
	    std::cerr<<"参数文件 parameter file "<<filename<<" 文件不存在 "<<std::endl;//文件不存在
	    config_->file_.release();//释放
	    return;
	}
    }
    //析构函数 退出函数  关闭文件
    Config::~Config()
    {
	if ( file_.isOpened() )// 若文件打开，
	    file_.release();//若文件打开，则 退出时 关闭文件
    }
    shared_ptr<Config> Config::config_ = nullptr;//初始化为 空

}
