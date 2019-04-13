// 使用GPU输出Hello world

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__global__ void helloFromGPU (void) 
{
    printf("Hello World from GPU!\n");
}

int main(void)
{
    // hello from cpu
    printf("Hello World from CPU!\n");

    for(int i=0;i<1000;i++)
    {
        helloFromGPU <<<1, 100>>>();
        cudaDeviceReset();
    }
    
    
    return 0;
}