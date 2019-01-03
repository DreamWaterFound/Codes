#include <iostream>
#include "glog/logging.h"
using namespace std;

//#define VERSION_1
#define VERSION_2


int main(int argc, char* argv[])
{

    //初始化 
    google::InitGoogleLogging(argv[0]);

#ifdef VERSION_1
    //设置日志存放目录
    FLAGS_log_dir="../log/";
#endif

#ifdef VERSION_2
    google::SetLogDestination(google::GLOG_FATAL, "../log/log_fatal_"); // 设置 google::FATAL 级别的日志存储路径和文件名前缀
    google::SetLogDestination(google::GLOG_ERROR, "../log/log_error_"); //设置 google::ERROR 级别的日志存储路径和文件名前缀
    google::SetLogDestination(google::GLOG_WARNING, "../log/log_warning_"); //设置 google::WARNING 级别的日志存储路径和文件名前缀
    google::SetLogDestination(google::GLOG_INFO, "../log/log_info_"); //设置 google::INFO 级别的日志存储路径和文件名前缀
#endif

    cout<<"Glog Test."<<endl;

    LOG(INFO)<<"Hello glog";
    LOG(WARNING)<<"Warning msg";
    LOG(ERROR)<<"error msg";
    //LOG(FATAL)<<"fatal msg";
    

    
    cout<<"Complete."<<endl;

    return 0;
}


