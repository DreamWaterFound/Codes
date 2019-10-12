/**
 * @file dummy.h
 * @author guoqing (1337841346@qq.com)
 * @brief 一个虚假的头文件
 * @version 0.1
 * @date 2019-10-12
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#ifndef __DUMMY_HPP__
#define __DUMMY_HPP__

#include <iostream>

void my_print(void)
{
    using std::cout;
    using std::endl;

    cout<<"::my_print() OK."<<endl;

    return ;
}

#endif  // macro __DUMMY_HPP__