#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "kernel.h"

using namespace cv;  

// GPU 核函数
__global__ void kernel(unsigned char *ptr, unsigned char ooo)  
{  
    // map from blockIdx to pixel position    
    // 获得每个线程处理的图像像素坐标  
    int x = threadIdx.x + blockIdx.x * blockDim.x;  
    int y = threadIdx.y + blockIdx.y * blockDim.y;  
    // 写图像指针的时候的偏移量
    int offset = x + y * blockDim.x * gridDim.x;  
  
    __shared__  float sharedMem[16][16];  
    const float period = 128.0f;  
    sharedMem[threadIdx.x][threadIdx.y] =  
        255 * 
        (sinf(x*2.0f*PI / period) + 1.0f) *  
        (sinf(y*2.0f*PI / period) + 1.0f) / 4.0f;  

    __syncthreads();     
  
    // 写通道颜色: BGR 但是注意, 这里当前线程读取的并不是自己操作的那个shared_memory的float变量!!!!!
    // 这就导致了不同warp(当前是10x10=100=32x3+4,4个warp)中必须进行线程同步
    ptr[offset * 3 + 0] = 0;
    ptr[offset * 3 + 1] = sharedMem[ooo - threadIdx.x][ooo - threadIdx.y];  
    ptr[offset * 3 + 2] = 0;


}

// 程序入口
void KernelLauncher(unsigned char ooo)
{
    // 图像指针
    DataBlock   data;  
    // CUDA 错误类型
    cudaError_t error;  
  
    // 创建一张 DIM x DIM 大小的图像， 填充黑色
    Mat image = Mat(DIM, DIM, CV_8UC3, Scalar::all(0));  
  
    // 显存指针
    unsigned char    *dev_bitmap;  
  
    // 在显存上分配指定图像大小的区域
    error = cudaMalloc((void**)&dev_bitmap, 3 * image.cols*image.rows);  
    // 保存设备端的位图指针
    data.dev_bitmap = dev_bitmap;  
  
    // 当前 Grid 中是 60 x 60 个 block
    dim3   grid(DIM / 10, DIM / 10);  
    // 每个 block 中是 10 x 10 个线程
    dim3   block(10, 10);  

    //DIM*DIM个线程块    
    kernel << <grid, block >> > (dev_bitmap, ooo);  
  
    // 将计算得到的结果
    error = cudaMemcpy(
        image.data,                     // 图像数据区, 主机端的
        dev_bitmap,                     // 显存指针, 设备端的
        3 * image.cols*image.rows,      // 拷贝长度, 就是图像占用的内存大小
        cudaMemcpyDeviceToHost);        // 拷贝方向
  
    // 然后释放之前占据的区域
    error = cudaFree(dev_bitmap);  
  
    // 工作结束, 显示结果
    imshow("__share__ and __syncthreads()", image);  
    waitKey(500);  

}
