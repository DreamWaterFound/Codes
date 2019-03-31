#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <sstream>

#include "emmintrin.h"
#include "tmmintrin.h"

// #include <stdint.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <intrin.h> 
// #include <>

using namespace std;


double addByNormalCPP(vector<float>& data1,vector<float>& data2, size_t loop_n, size_t loop_res);
double addBySSE(vector<float>& data1,vector<float>& data2, size_t loop_n, size_t loop_res);


int main(int argc, char* argv[])
{
    cout<<"SSE Demo."<<endl;

    if(argc!=4)
    {
        cout<<"Usage: "<<argv[0]<<" data_path time_log_path iteration"<<endl;
        return 0;
    }

    ifstream ifs;
    ifs.open(argv[1]);
    if(!ifs)
    {
        cout<<"Fatal Error: Data file "<<argv[1]<<" open failed. Please check it."<<endl;
        return 0;
    }

    ofstream ofs;
    ofs.open(argv[2]);
    if(!ofs)
    {
        cout<<"Fatal Error: Log file "<<argv[2]<<" open failed. Please check it."<<endl;
        return 0;
    }
    ofs<<"SSE demo time-cost log file."<<endl;

    size_t iter_num;
    stringstream ss(argv[3]);
    ss>>iter_num;
    if(iter_num==0)
    {
        cout<<"Warning: Iter_num set to 0 and the program terminated."<<endl;
        return 0;
    }

    size_t n;
    double upper,lower;
    ifs>>n>>upper>>lower;

    cout<<"Info: Open file "<<argv[1]<<" successed, the number of data is "<<n<<", upper bound is "<<upper<<" and the lower bound is "<<lower<<"."<<endl;
    ofs<<"number of data = "<<n<<" , upper bound = "<<upper<<" , lower bound = "<<lower<<endl;

    size_t loop_n=n/(4);
    size_t loop_res=(n-4*loop_n);

    cout<<"Info: loop_n = "<<loop_n<<", loop_res = "<<loop_res<<endl;
    cout<<"Loading data file, waiting ..."<<endl;

    //读取所有数据到内存中
    vector<float> data1,data2;
    float d;
    for(size_t i=0;i<n;++i)
    {
        ifs>>d;
        data1.push_back(d);
        ifs>>d;
        data2.push_back(d);
    }
    ifs.close();
    cout<<"Done."<<endl;
    cout<<"sizeof(float) = "<<sizeof(float)<<endl;

    ofs<<"iter_num = "<<iter_num<<endl;
    cout<<"iter_num = "<<iter_num<<endl;
    
    ofs.precision(0);
    ofs.setf(std::ios::fixed, std::ios::floatfield);

    for(size_t i=0; i<iter_num;++i)
    {
        //正常的 C++ 处理
        cout<<"Iter = "<<i+1<<"  Computing by normal C++ method, waiting ..."<<endl;
        
        double t1=addByNormalCPP(data1,data2,loop_n,loop_res);
        cout.precision(4);
        cout.setf(std::ios::fixed, std::ios::floatfield);
        cout<<"Complete, time = "<<t1<<" ns"<<endl;

        //使用SSE加速
        cout<<"Iter = "<<i+1<<"  Computing by SSE method, waiting ..."<<endl;
        double t2=addBySSE(data1,data2,loop_n,loop_res);
        cout.precision(4);
        cout.setf(std::ios::fixed, std::ios::floatfield);
        cout<<"Complete, time = "<<t2<<" ns"<<endl;
        

        ofs<<t1<<"\t"<<t2<<endl;
    }

    
    

    

/*

    float op1[4] __attribute__((aligned(16))) = {1.0, 2.0, 3.0, 4.0};  
    float op2[4] __attribute__((aligned(16))) = {1.0, 2.0, 3.0, 4.0};  
    float result[4] __attribute__((aligned(16))) ;       
  
    __m128  a;  
    __m128  b;  
    __m128  c;  
  
    // Load  
    a = _mm_load_ps(op1);  
    b = _mm_load_ps(op2);  
  
    // Calculate  
    c = _mm_add_ps(a, b);   // c = a + b  
  
    // Store  
    _mm_store_ps(result, c);  
  
    
    printf("0: %lf\n", result[0]);  
    printf("1: %lf\n", result[1]);  
    printf("2: %lf\n", result[2]);  
    printf("3: %lf\n", result[3]);  
  */
    ofs.flush();
    ofs.close();


    return 0;  
}

double addByNormalCPP(vector<float>& data1,vector<float>& data2, size_t loop_n, size_t loop_res)
{
    std::chrono::duration<double, nano> duration;
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();

    //volatile float res1,res2,res3,res4;
    float res1,res2,res3,res4;
    size_t base;
    for(size_t i=0;i<loop_n;++i)
    {
        base=i<<2;
        res1=data1[base+0] + data2[base+0];
        res2=data1[base+1] + data2[base+1];
        res3=data1[base+2] + data2[base+2];
        res4=data1[base+3] + data2[base+3];

        //虚假操作，避免编译器误优化
        //res1=res1+res2+res3+res4;
    }
/*
    base=loop_n<<2;
    for(size_t i=0;i<loop_res;++i)
    {
        res1=data1[base+i]+data2[base+i];
    }
*/
    //计算完成，停止计时
    duration=std::chrono::steady_clock::now()-start;
    return duration.count();
}

double addBySSE(vector<float>& data1,vector<float>& data2, size_t loop_n, size_t loop_res)
{
    std::chrono::duration<double, nano> duration;
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();

    float op1[4]        __attribute__((aligned(16))) ;  
    float op2[4]        __attribute__((aligned(16))) ; 
    float result[4]     __attribute__((aligned(16))) ; 

    //volatile float res1,res2,res3,res4;
    size_t base;
    volatile __m128  a;  
    volatile __m128  b;  
    volatile __m128  c;
    for(size_t i=0;i<loop_n;++i)
    {
        base=i<<2;

        //perpare
        op1[0]=data1[base+0];
        op1[1]=data1[base+1];
        op1[2]=data1[base+2];
        op1[3]=data1[base+3];

        op2[0]=data2[base+0];
        op2[1]=data2[base+1];
        op2[2]=data2[base+2];
        op2[3]=data2[base+3];

        a = _mm_load_ps(op1);  
        b = _mm_load_ps(op2);

        c = _mm_add_ps(a, b);

        _mm_store_ps(result, c); 

        //虚假操作，避免编译器误优化
        //res1=res1+res2+res3+res4;
    }
/*
    base=loop_n<<2;
    for(size_t i=0;i<loop_res;++i)
    {
        res1=data1[base+i]+data2[base+i];
    }
*/
    //计算完成，停止计时
    duration=std::chrono::steady_clock::now()-start;
    return duration.count();
}