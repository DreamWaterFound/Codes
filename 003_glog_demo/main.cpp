#include <iostream>
#include "glog/logging.h"
using namespace std;

int main(int argc, char* argv[])
{
    //初始化 
    google::InitGoogleLogging(argv[0]);

    //设置日志存放目录
    FLAGS_log_dir="./log/";

    cout<<"Glog Test."<<endl;

    LOG(INFO)<<"Hello glog";
    LOG(WARNING)<<"Warning msg";
    LOG(ERROR)<<"error msg";
    //LOG(FATAL)<<"fatal msg";
    

    
    cout<<"Complete."<<endl;

    return 0;
}


