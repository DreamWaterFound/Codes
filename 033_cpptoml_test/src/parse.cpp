/**
 * @file parse.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 一个简单的解析展示
 * @version 0.1
 * @date 2019-07-03
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "tools/cpptoml.h"

#include <iostream>
#include <cassert>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " filename" << std::endl;
        return 1;
    }

    try
    {
        // 类的静态成员函数
        std::shared_ptr<cpptoml::table> g = cpptoml::parse_file(argv[1]);
        // 输出配置文件的内容
        std::cout << (*g) << std::endl;
    }
    catch (const cpptoml::parse_exception& e)
    {
        std::cerr << "Failed to parse " << argv[1] << ": " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
