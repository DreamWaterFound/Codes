/**
 * @file main.c
 * 实现程序的主要功能
 * 
 * @author Guoqing Liu
 * @version V1.0
 * @date 2018.12.24
 * @copyright WTFPL V2,Dec,2004
 * 
*/

///包含必要的头文件
#include <iostream>
#include "classA.h"
#include "classB.h"

//使用标准名字空间
using namespace std;

<<<<<<< HEAD
//这个文件是主文件。

=======
/**
 * @brief 主函数
 * 
 * @param argc 参数的个数
 * @param argv 参数队列
 * @return int 返回值恒为0，表示程序正常退出。
 */
>>>>>>> add_doxygen_2
int main(int argc,char *argv[])
{
    cout<<"Doxygen Demo."<<endl;

    A obj(2);
    B objb(1);

    ///这里什么都不做就直接进行析构了。
    cout<<"准备析构。"<<endl;


    return 0;
}