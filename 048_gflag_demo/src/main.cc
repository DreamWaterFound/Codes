#include <iostream>
#include "gflags/gflags.h"

using std::cout;
using std::endl;

// 对参数名称进行初始化
DEFINE_string(confPath, "test.conf", "program configure file.");
DEFINE_int32(port, 1111, "program listen port");
DEFINE_bool(debug, true, "run debug mode");

int main(int argc, char** argv)
{
    cout<<"Test text."<<endl;
    cout<<"Complied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    // 这个需要放在解析符号之前
    gflags::SetVersionString("1.1");
    // 解析符号
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // 输出当前给定的命令行参数
    cout << "argc= " << argc << endl;
    for(int i = 0; i<argc; ++i)
        cout << "argv[" << i << "]:" << argv[i] << endl;;

    // 输出解析得到的数据
    cout << "confPath = " << FLAGS_confPath << endl;
    cout << "port = " << FLAGS_port << endl;

    // 对当前是否是DEBUG模式的判断
    if (FLAGS_debug) {
    cout << "this is a debug mode..." << endl;
    }
    else {
    cout << "this is a nodebug mode...." << endl;
    }

    cout << "ohhhhh succuss~~" << endl;

    // 析构
    gflags::ShutDownCommandLineFlags();

    return 0;
}           