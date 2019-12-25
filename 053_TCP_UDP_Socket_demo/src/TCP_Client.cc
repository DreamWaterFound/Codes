// TODO 完成了工作原理分析之后, 将这里的头文件依次注释掉, 看看哪个头文件提供了哪个函数
#include <stdio.h>              // C 基本io操作
#include <errno.h>              // 错误码相关的宏
#include <string.h>             // c 字符串支持
#include <netdb.h>              // Linux 系系统中定义和网络操作相关操作和数据类型的头文件
#include <sys/types.h>          // 定义了 Linux 系统中常使用的基本数据类型
#include <netinet/in.h>         // 和计算机中的网络概念相关
#include <sys/socket.h>         // 套接字支持
#include <stdlib.h>             // c 标准库头文件
#include <unistd.h>             // 一些系统API
#include <arpa/inet.h>          // 网络相关,似乎和通信协议有关
#include <netdb.h>              // Linux 中定义了和网络的有关信息

// ?
#define MAXLINE 4096

// 主函数
int main(int argc, char** argv)
{
    // step 0 数据准备
    // 套接字的 file id
    int   sockfd;
    // ?
    int   len;
    // ? 消息的缓冲区
    char  buffer[MAXLINE];
    // 套接字地址
    struct sockaddr_in  servaddr;

    // 文件指针(C语言风格)
    FILE *fq;

    // 输入参数检查
    if( argc != 2)
    {
        // 要求第一个命令行参数是服务器的ip地址
        printf("usage: ./client <ip_address>\n");
        return 0;
    }

    // step 1 创建套接字
    sockfd = socket(
        AF_INET,            // 网络的域, 这里直接使用了" IP protocol family "
        SOCK_STREAM,        // 套接字的类型, SOCK_STREAM = 无连接、不可靠的固定最大长度数据报
        0);                 // 选择的通信协议, 设置为 0 表示默认选择通信协议
    // 检查是否正确生成
    if( sockfd < 0 )
    {
        // 说明套接字创建失败
        printf("create socket error: %s(errno: %d)\n", 
                strerror(errno),    // strerror 可以获取 errno 错误码的字符串描述
                errno);             // errno 为一个宏, 获取最近的错误码
        return 0;
    }

    // step 2 设置套接字?
    // 全部初始化为默认值
    memset(&servaddr, 0, sizeof(servaddr));
    // 域
    servaddr.sin_family = AF_INET;
    // 端口
    servaddr.sin_port = htons(6666);
    // 将命令行参数中给出的文本形式的 ip 地址转换成为 uint32_t 的形式并且保存到 servaddr.sin_addr 中
    if( inet_pton(AF_INET, argv[1], &servaddr.sin_addr) <= 0){
        printf("inet_pton error for %s\n",argv[1]);
        return 0;
    }

    // 使用套接字对象和套接字配置, 连接远程程序
    if( connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0){
        printf("connect error: %s(errno: %d)\n",strerror(errno),errno);
        return 0;
    }

    // HACK 现在我们先假设是在项目的 bin 目录下执行
    if( ( fq = fopen("../data/send/test_pic.jpg","rb") ) == NULL ){
        // 文件打开失败
        printf("File open failed.\n");
        // 关闭套接字
        close(sockfd);
        exit(1);
    }

    // 初始值为0的缓冲区(算不上是开辟, 只能说是在现有的数组基础上进行了0值初始化)
    bzero(buffer,sizeof(buffer));

    // 
    while(!feof(fq)){
        
        // 每次先读写 buffer 等同大小的数据再发送. 返回的len是实际读写的大小.
        len = fread(buffer, 1, sizeof(buffer), fq);

        // DEBUG
        printf("len = %d \n",len);

        // 理论上写入到套接字中的数据量应该是相同的
        if(len != write(sockfd, buffer, len)){
            // 直接跳出发送数据的过程
            printf("write failed.\n");
            break;
        }
    }

    // 传输结束, 关闭套接字, 关闭读写的文件
    close(sockfd);
    fclose(fq);

    return 0;
}
