// 必要的头文件
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

// 数组元素相加
__global__ void simple_add(const int *A, const int *B, int *C)
{
	C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

// 主函数
int main(int arg, char* args[])
{
    // 待操作的数组
	const int size = 10;
	int A[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int B[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };
	int C[size];

    // 显存中数据的指针
	int *buffer_A = 0;
	int *buffer_B = 0;
	int *buffer_C = 0;
	cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
	{
		std::cout << "No CUDA devices found!" << std::endl;
		exit(1);
    }

    // 获取设备名
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	std::cout << "Using device: " << prop.name << std::endl;
	
    // Allocate GPU buffers for three vectors (two input, one output).
	cudaMalloc((void**)&buffer_A, size * sizeof(int));
	cudaMalloc((void**)&buffer_B, size * sizeof(int));
	cudaMalloc((void**)&buffer_C, size * sizeof(int));
	
    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(buffer_A, A, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(buffer_B, B, size * sizeof(int), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU with one thread for each element.
    simple_add<<<1, size>>>(buffer_A, buffer_B, buffer_C);

    // Check for any errors launching the kernel
    // NOTICE 这个是可以学习的，在执行完程序之后，判断核函数的执行状态
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
	{
		std::cout << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaFree(buffer_A);
		cudaFree(buffer_B);
		cudaFree(buffer_C);
		exit(1);
    }
    
    //? 提问，那么为什么不是在同步状态之后再对核函数的执行状态进行确认呢?
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
	{
        // 出错啦，直接释放相关的内存区域
		std::cout << "Could not synchronize device!" << std::endl;
		cudaFree(buffer_A);
		cudaFree(buffer_B);
		cudaFree(buffer_C);
		exit(1);
    }
	
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(C, buffer_C, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(buffer_A);
	cudaFree(buffer_B);
	cudaFree(buffer_C);

    // NOTICE 时刻判断程序的运行结果
	if(cudaStatus != cudaSuccess)
	{
		std::cout << "Could not copy buffer memory to host!" << std::endl;
		exit(1);
	}

    //Prints the array
	std::cout << "Result:" << std::endl;
	for (int i = 0; i < size; i++)
	{
		std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
	{
		std::cout << "Device reset failed!" << std::endl;
        exit(1);
    }

    return 0;
}