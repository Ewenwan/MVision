/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
private://私有对象
    static std::shared_ptr<Config> config_; //共享指针 把 智能指针定义成 Config的指针类型 以后参数传递时  使用Config:Ptr 类型就可以了
    cv::FileStorage file_;//使用 opencv提供的FileStorage类 读取一个YAML文件  可以通过模板函数get 获取任意类型的参数值
    
    Config () {} // private constructor makes a singleton
public:
    ~Config();  // close the file when deconstructing 
    
    // 读取一个参数文件 set a new config file 
    static void setParameterFile( const std::string& filename ); 
    
    // access the parameter values
    //获取参数的值
    template< typename T >
    static T get( const std::string& key )//可以通过模板函数get 获取任意类型的参数值
    {
        return T( Config::config_->file_[key] );
    }
};
}

#endif // CONFIG_H
