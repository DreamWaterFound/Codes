// 使用线程块和线程的概念，在 GPU 上进行向量相加的并行计算
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>


__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx]; // 检查是否越界
    printf("%f + %f = %f Caculated On GPU: block %d thread %d\n", 
            A[idx], B[idx], C[idx], blockIdx.x, threadIdx.x);
}

void initialData(float *ip, int size){
    // generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i=0; i<size; i++){
        ip[i] = (float)(rand() & 0xFF)/10.0f;
    }
}

void print(float *array, const int N){
    for (int idx=0; idx<N; idx++){
        printf(" %f", array[idx]);
    }
    printf("\n");
}

// 主函数
int main(){
    int nElem = 6;
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B;

    // 在hosts上分配空间
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    // 获得随机的初始数据
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    printf("向量 A:");
    print(h_A, nElem);
    printf("向量 B:");
    print(h_B, nElem);


    // 在GPU端分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    // 数据传递
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
   
    printf("向量 C 的每个元素计算过程:\n");
    // 啥骚操作？
    dim3 block(2);
    dim3 thread(3);
    sumArraysOnGPU <<< block, thread >>>(d_A, d_B, d_C, nElem); // 异步计算

    // 记得释放
    free(h_A);
    free(h_B);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}