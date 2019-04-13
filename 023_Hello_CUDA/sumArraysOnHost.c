// sumArraysOnHost.c
// 在CPU上进行数据求和的计算
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// 在主机上执行相加运算，前三个参数分别是数组的收地址，第四个参数保存了这三个数组的长度
void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for (int idx=0; idx<N; idx++){
        C[idx] = A[idx] + B[idx];
    }
}

// 初始化数据值，参数1是数组的首地址，参数2是数组的长度
void initialData(float *ip, int size){
    // generate different seed for random number
    // 产生随机数种子
    time_t t;
    srand((unsigned int) time(&t));

    // 使得这个数组产生随机的内容
    for (int i=0; i<size; i++){
        ip[i] = (float)(rand() & 0xFF)/10.0f;
    }
}

// 打印数组，参数1是数组的首地址，参数2是数组的长度
void print(float *array, const int N){
    // 分别显示数组中的每一个元素
    for (int idx=0; idx<N; idx++){
        printf(" %f", array[idx]);
    }
    printf("\n");
}

//主函数
int main(){
    // 每个数组有四个元素
    int nElem = 4;
    // 计算这样的数据会占用多少内存空间
    size_t nBytes = nElem * sizeof(float);
    // 保存数组的地址的指针
    float *h_A, *h_B, *h_C;
    // 分配这几个数组用到的存储空间
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    // 获得初始的随机数据
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    print(h_A, nElem);
    print(h_B, nElem);

    // 在CPU上进行数组的加和计算
    sumArraysOnHost(h_A, h_B, h_C, nElem);
    print(h_C, nElem);
    // 记得释放
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}