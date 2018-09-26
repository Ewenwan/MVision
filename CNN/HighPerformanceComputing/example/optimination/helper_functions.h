// 计时函数 
#pragma once // 只编译一次
#include <string>
#include <chrono>
#include <iostream>
#include <vector>

//////////////////////////////////////////////////////////////////////////
//scope exit
namespace
{
	template <typename FuncType>
	class InnerScopeExit
	{
	public:
		InnerScopeExit(const FuncType _func) :func(_func){}
		~InnerScopeExit(){ if (!dismissed){ func(); } }
	private:
		FuncType func;
		bool dismissed = false;
	};
	template <typename F>
	InnerScopeExit<F> MakeScopeExit(F f) 
	{
		return InnerScopeExit<F>(f);
	};
}
#define DO_STRING_JOIN(arg1, arg2) arg1 ## arg2
#define STRING_JOIN(arg1, arg2) DO_STRING_JOIN(arg1, arg2)
#define SCOPEEXIT(code) auto STRING_JOIN(scope_exit_object_, __LINE__) = ::MakeScopeExit([&](){code;});

//////////////////////////////////////////////////////////////////////////
//function cost time calculate helper
class FuncCostTimeHelper
{
public:
	FuncCostTimeHelper(const std::string& _tag) :tag(_tag)
	{
		start_time = std::chrono::high_resolution_clock::now();
	}
	~FuncCostTimeHelper()
	{
		stop_time = std::chrono::high_resolution_clock::now();
		const auto cost_time = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count(); //us
		std::cout << tag << " cost time : " << cost_time << " us" << std::endl;
	}
private:
	std::string tag;
	std::chrono::high_resolution_clock::time_point start_time;
	std::chrono::high_resolution_clock::time_point stop_time;
};
