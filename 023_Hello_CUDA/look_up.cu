// 程序功能：查看当前服务器上具有的显卡数目，并且分别获取他们的详细属性。

// 必要的CUDA 包含文件
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 传统 C++ 流输入输出支持
#include <iostream>
 
using namespace std;
 
// 主函数
int main()
{
	// 设备属性的变量
	cudaDeviceProp deviceProp;
	// 设备计数
	int deviceCount;
	// 保存调用函数的输出结果
	cudaError_t cudaError;
	// 获取当前的设备总数
	cudaError = cudaGetDeviceCount(&deviceCount);
	cout<<"We have "<<deviceCount<<" device(s)."<<endl;
	// 获得每一个设备的属性
	for (int i = 0; i < deviceCount; i++)
	{
		// 获得属性
		cudaError = cudaGetDeviceProperties(&deviceProp, i);
 
		cout << "设备 " << i + 1 << " 的主要属性： " << endl;
		cout << "设备显卡型号： " << deviceProp.name << endl;
		cout << "设备全局内存总量（以MB为单位）： " << deviceProp.totalGlobalMem / 1024 / 1024 << endl;
		cout << "设备上一个线程块（Block）中可用的最大共享内存（以KB为单位）： " << deviceProp.sharedMemPerBlock / 1024 << endl;
		cout << "设备上一个线程块（Block）中可用的32位寄存器数量： " << deviceProp.regsPerBlock << endl;
		cout << "设备上一个线程块（Block）可包含的最大线程数量： " << deviceProp.maxThreadsPerBlock << endl;
		cout << "设备的计算功能集（Compute Capability）的版本号： " << deviceProp.major << "." << deviceProp.minor << endl;
		cout << "设备上多处理器的数量： " << deviceProp.multiProcessorCount << endl;
	}

	return 0;
}
