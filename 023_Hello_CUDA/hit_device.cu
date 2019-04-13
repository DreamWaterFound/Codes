// 这个程序选中符合指定条件的设备

// CUDA 支持
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 传统 C++ 支持
#include <iostream>
 
using namespace std;
 
// 主函数，还没有输入参数
int main()
{
	//定义需要的设备属性
	cudaDeviceProp devicePropDefined;
    memset(&devicePropDefined, 0, sizeof(cudaDeviceProp));  //设置devicepropDefined的值
    // 版本号的要求
	devicePropDefined.major = 5;
	devicePropDefined.minor = 2;
 
	int devicedChoosed;  //选中的设备ID
	cudaError_t cudaError;
	cudaGetDevice(&devicedChoosed);  //获取当前设备ID
	cout << "当前使用设备的编号： " << devicedChoosed << endl;
 
	cudaChooseDevice(&devicedChoosed, &devicePropDefined);  //查找符合要求的设备ID
	cout << "满足指定属性要求的设备的编号： " << devicedChoosed << endl;
 
	cudaError = cudaSetDevice(devicedChoosed); //设置选中的设备为下文的运行设备
 
	if (cudaError == cudaSuccess)
		cout << "设备选取成功！" << endl;
	else
        cout << "设备选取失败！" << endl;
        
    char c;
    cin>>c;
	return 0;
}
