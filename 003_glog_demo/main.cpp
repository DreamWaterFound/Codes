#include <iostream>
#include "glog/logging.h"
using namespace std;

//#define VERSION_1
#define VERSION_2


int main(int argc, char* argv[])
{

    // ref https://www.cnblogs.com/hiloves/p/6009707.html

    // 生成日志文件的时候不输出前缀
    // FLAGS_log_prefix = false;
    // 避免在磁盘接近满时输出
    FLAGS_stop_logging_if_full_disk = true;
    // 如果输出到终端上则要添加颜色
    FLAGS_colorlogtostderr = true;
    
    // 大于该等级的日志信息均输出到屏幕上
    google::SetStderrLogging(google::GLOG_INFO);


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

    // 设置生成日志的后缀,但是感觉好像不起作用
    google::SetLogFilenameExtension("log_");

#endif

    cout<<"Glog Test."<<endl;

    LOG(INFO)<<"Hello glog";
    LOG(WARNING)<<"Warning msg";
    LOG(ERROR)<<"error msg";
    //LOG(FATAL)<<"fatal msg";
    

    
    cout<<"Complete."<<endl;

    google::ShutdownGoogleLogging();

    return 0;
}


