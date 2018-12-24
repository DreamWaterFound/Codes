/**
 * @file classB.cpp
 * @author Guoqing Liu (1337841346@qq.com)
 * @brief 类B的实现文件
 * @version 0.1
 * @date 2018-12-24
 * 
 * @copyright WTFPL V2, Dec, 2004
 * 
 */


#include "classB.h"

/**
 * @brief B类对象的无参数构造函数
 * 
 */
B::B()
{
    ///输出调试信息
    cout<<"类B的无参数构造函数被调用。"<<endl;
    mnNum=0;
}

/**
 * @brief B类对象的有参数构造函数
 * 
 * @param n 成员变量的值
 */
B::B(int n)
{
    cout<<"类B的有参数构造函数被调用，参数值为"<<n<<endl;
    mnNum=n;
}

/**
 * @brief B类对象的析构函数
 * 
 */
B::~B()
{
    cout<<"类B的析构函数被调用。"<<endl;
}


/**
 * @brief 设备对象的值
 * 
 * @param n 对象的值
 */
void B::setNum(int n)
{
    mnNum=n;
}

/**
 * @brief 获得对象的值
 * 
 * @return int 返回对象的值
 */
int B::getNum(void)
{
    return mnNum;
}
