#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <ctime>
using namespace std;

double RangedRandDemo( double range_min, double range_max);


int main(int argc, char* argv[])
{
    cout<<"Generate Data Tool."<<endl;

    //检查参数
    if(argc!=5)
    {
        cout<<"Usage: "<<argv[0]<<" data_file_path data_num data_range_upper_bound data_range_lower_bound "<<endl;
        return 0;
    }

    //打开文件
    ofstream ofs;
    ofs.open(argv[1]);
    if(!ofs)
    {
        cout<<"Error: File "<<argv[1]<<" can not be openned! "<<endl;
        return 0;
    }

    size_t nData;
    double dUpper,dLower;
    stringstream ss(argv[2]);
    ss>>nData;
    ss=stringstream(argv[3]);
    ss>>dUpper;
    ss=stringstream(argv[4]);
    ss>>dLower;

    // 这里使用等于号其实不严格
    if(dUpper<=dLower)
    {
        cout<<"Error: data_range_upper_bound = "<<dUpper<<" < data_range_lower_bound = "<<dLower<<" !"<<endl;
        return 0;
    }

    cout<<"Check Ok, please waiting ..."<<endl;

    //数据文件格式： 第一行特殊，为 数据个数 + 数据上界 + 数据下界
    ofs<<nData<<" "<<dUpper<<" "<<dLower<<endl;

    //接下来几行就都是产生的随机数了, 是数对的形式存储的
    std::srand(std::time(NULL));
    for(size_t i=0;i<nData;++i)
    {
        ofs<<RangedRandDemo(dUpper,dLower)<<" "<<RangedRandDemo(dUpper,dLower)<<endl;
    }


    ofs.flush();
    ofs.close();

    cout<<"Done!"<<endl;

    return 0;
}

#ifndef RAND_MAX
#define RAND_MAX 32767
#endif

// 来自MSDN
double RangedRandDemo( double range_min, double range_max)
{
   // Generate random numbers in thehalf-closed interval
   // [range_min, range_max). In other words,
   // range_min <= random number <range_max
  
    return (double)rand() / (RAND_MAX + 1.0) *(range_max - range_min) // RAND_MAX = 32767
            + range_min;
}
