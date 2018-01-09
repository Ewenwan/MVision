/*
常用 头文件 库包含
 */


#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H

// define the commonly included file to avoid a long include list
// for Eigen  矩阵 姿态四元数等几何学描述库
#include <Eigen/Core>
#include <Eigen/Geometry>
using Eigen::Vector2d;//二维点
using Eigen::Vector3d;//三维点

// for Sophus 李群李代数 库
#include "sophus/se3.h"
using Sophus::SE3;

// for cv opencv
#include <opencv2/core/core.hpp>
using cv::Mat;

// std  标志库头文件
#include <vector>//容器  封装数组    存储为 连续内存单元 支持  [] 操作   尾部插入速度很快
#include <list>//列表       封装了链表  存储单元不连续    随机增减内容 快
#include <memory>//存储
#include <string>//字符串
#include <iostream>//输入输出流
#include <set>// Set只含有Key   set 是一个容器，它用于储存数据并且能从一个数据集合中取出数据。它的每个元素的值
//必须惟一，而且系统会根据该值来自动将数据排序。每个元素的值不能直接被改变。
// vector封装数组，list封装了链表，map和set封装了二叉树等

#include <unordered_map>// 无序   Key － value的对应
// unordered_map和map类似，都是存储的key-value的值，可以通过key快速索引到value。不同的是unordered_map不会根据key的大小进行排序，
// 存储时是根据key的hash值判断元素是否相同，即unordered_map内部元素是无序的，而map中的元素是按照二叉搜索树存储，进行中序遍历会得到有序遍历。
// Map和Hash_Map的区别是Hash_Map使用了Hash算法来加快查找过程，但是需要更多的内存来存放这些Hash桶元素，因此可以算得上是采用空间来换取时间策略。


#include <map>// Key － value的对应  字典 封装了二叉树 map内部自建一颗红黑树(一 种非严格意义上的平衡二叉树)
// 这颗树具有对数据自动排序的功能，所以在map内部所有的数据都是有序的
// 自动建立Key － value的对应。key 和 value可以是任意你需要的类型。
//std:map<int,string> personnel;  这样就定义了一个用int作为索引,并拥有相关联的指向string的指针.
//map<int, string> mapStudent;  
//mapStudent.insert(pair<int, string>(1, "student_one"));  


using namespace std; 
#endif

/*
 set的各成员函数列表如下:
1. begin()--返回指向第一个元素的迭代器
2. clear()--清除所有元素
3. count()--返回某个值元素的个数
4. empty()--如果集合为空，返回true
5. end()--返回指向最后一个元素的迭代器
6. equal_range()--返回集合中与给定值相等的上下限的两个迭代器
7. erase()--删除集合中的元素
8. find()--返回一个指向被查找到元素的迭代器
9. get_allocator()--返回集合的分配器
10. insert()--在集合中插入元素
11. lower_bound()--返回指向大于（或等于）某值的第一个元素的迭代器
12. key_comp()--返回一个用于元素间值比较的函数
13. max_size()--返回集合能容纳的元素的最大限值
14. rbegin()--返回指向集合中最后一个元素的反向迭代器
15. rend()--返回指向集合中第一个元素的反向迭代器
16. size()--集合中元素的数目
17. swap()--交换两个集合变量
18. upper_bound()--返回大于某个值元素的迭代器
19. value_comp()--返回一个用于比较元素间的值的函数


1、map简介

map是一类关联式容器。它的特点是增加和删除节点对迭代器的影响很小，除了那个操作节点，对其他的节点都没有什么影响。对于迭代器来说，可以修改实值，而不能修改key。

2、map的功能

自动建立Key － value的对应。key 和 value可以是任意你需要的类型。
根据key值快速查找记录，查找的复杂度基本是Log(N)，如果有1000个记录，最多查找10次，1,000,000个记录，最多查找20次。
快速插入Key - Value 记录。
快速删除记录
根据Key 修改value记录。
遍历所有记录。
3、使用map

使用map得包含map类所在的头文件

#include <map> //注意，STL头文件没有扩展名.h

map对象是模板类，需要关键字和存储对象两个模板参数：

std:map<int, string> personnel;

这样就定义了一个用int作为索引,并拥有相关联的指向string的指针.

为了使用方便，可以对模板类进行一下类型定义，

typedef map<int, CString> UDT_MAP_INT_CSTRING;

UDT_MAP_INT_CSTRING enumMap;

4、在map中插入元素

改变map中的条目非常简单，因为map类已经对[]操作符进行了重载

enumMap[1] = "One";

enumMap[2] = "Two";



这样非常直观，但存在一个性能的问题。插入2时,先在enumMap中查找主键为2的项，没发现，然后将一个新的对象插入enumMap，键是2，值是一个空字符串，插入完成后，将字符串赋为"Two"; 该方法会将每个值都赋为缺省值，然后再赋为显示的值，如果元素是类对象，则开销比较大。我们可以用以下方法来避免开销：

enumMap.insert(map<int, CString> :: value_type(2, "Two"))

5、查找并获取map中的元素

下标操作符给出了获得一个值的最简单方法：

CString tmp = enumMap[2];

但是,只有当map中有这个键的实例时才对，否则会自动插入一个实例，值为初始化值。

我们可以使用Find()和Count()方法来发现一个键是否存在。

查找map中是否包含某个关键字条目用find()方法，传入的参数是要查找的key，在这里需要提到的是begin()和end()两个成员，分别代表map对象中第一个条目和最后一个条目，这两个数据的类型是iterator.

int nFindKey = 2; //要查找的Key

//定义一个条目变量(实际是指针)

UDT_MAP_INT_CSTRING::iterator it= enumMap.find(nFindKey);

if(it == enumMap.end()) {

//没找到

}

else {

//找到

}

通过map对象的方法获取的iterator数据类型是一个std::pair对象，包括两个数据 iterator->first 和 iterator->second 分别代表关键字和存储的数据

6、从map中删除元素

移除某个map中某个条目用erase()

该成员方法的定义如下

iterator erase(iterator it); //通过一个条目对象删除
iterator erase(iterator first, iterator last); //删除一个范围
size_type erase(const Key& key); //通过关键字删除
clear()就相当于 enumMap.erase(enumMap.begin(), enumMap.end());

7、map的基本操作函数：
      C++ Maps是一种关联式容器，包含“关键字/值”对
      begin()          返回指向map头部的迭代器
      clear(）         删除所有元素
      count()          返回指定元素出现的次数
      empty()          如果map为空则返回true
      end()            返回指向map末尾的迭代器
      equal_range()    返回特殊条目的迭代器对
      erase()          删除一个元素
      find()           查找一个元素
      get_allocator()  返回map的配置器
      insert()         插入元素
      key_comp()       返回比较元素key的函数
      lower_bound()    返回键值>=给定元素的第一个位置
      max_size()       返回可以容纳的最大元素个数
      rbegin()         返回一个指向map尾部的逆向迭代器
      rend()           返回一个指向map头部的逆向迭代器
      size()           返回map中元素的个数
      swap()            交换两个map
      upper_bound()     返回键值>给定元素的第一个位置
      value_comp()      返回比较元素value的函数

例子：

//遍历：

map<string,CAgent>::iterator iter;
 for(iter = m_AgentClients.begin(); iter != m_AgentClients.end(); ++iter)
 {
   if(iter->first=="8001"  {
   　　this->SendMsg(iter->second.pSocket,strMsg);//iter->first
   }
 }

//查找：

map<string,CAgent>::iterator iter=m_AgentClients.find(strAgentName);
 if(iter!=m_AgentClients.end())//有重名的  {
 }
 else //没有{
 }

//元素的个数

if (m_AgentClients.size()==0)

//删除

map<string,CAgent>::iterator iter=m_AgentClients.find(pSocket->GetName());
 if(iter!=m_AgentClients.end())
 {

     m_AgentClients.erase(iter);//列表移除
 }
 */
