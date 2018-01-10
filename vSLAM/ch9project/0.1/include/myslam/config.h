/*
 * myslam::Config
 *
 */
/*提供配置参数 */
#ifndef CONFIG_H //防止头文件重复引用
#define CONFIG_H//宏定义

#include "myslam/common_include.h" 

namespace myslam //命令空间下 防止定义的出其他库里的同名函数
{
class Config
{
private://私有对象 变量
    static std::shared_ptr<Config> config_; //共享指针 把 智能指针定义成 Config的指针类型 以后参数传递时  使用Config:Ptr 类型就可以了
    cv::FileStorage file_;//使用 opencv提供的FileStorage类 读取一个YAML文件  可以通过模板函数get 获取任意类型的参数值
   //【1】默认构造函数
    Config () {} // private constructor makes a singleton
public:
  // 【2】析构函数
    ~Config();  // close the file when deconstructing 
    
    // 读取一个参数文件 set a new config file 
    static void setParameterFile( const std::string& filename ); 
    
    // access the parameter values
    //获取参数的值
    template< typename T >// 模板变量  任意变量类型
    static T get( const std::string& key )//可以通过模板函数get 获取任意类型的参数值
    {
        return T( Config::config_->file_[key] );
    }
};
}

#endif // CONFIG_H
