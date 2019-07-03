/**
 * @file build_toml.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 这个是程序生成 toml 文件的实例,不过最后还是通过终端直接显示出来了
 * @version 0.1
 * @date 2019-07-03
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include <tools/cpptoml.h>
#include <iostream>

/** @brief 主函数 */
int main(int argc, char* argv[])
{
    // 生成一个空表,即默认不属于任何表,自己也不带有名字的默认表
    std::shared_ptr<cpptoml::table> root = cpptoml::make_table();
    // 基本数据类型的添加
    root->insert("Integer", 1234L);
    root->insert("Double", 1.234);
    root->insert("String", std::string("ABCD"));

    auto table = cpptoml::make_table();
    table->insert("ElementOne", 1L);
    table->insert("ElementTwo", 2.0);
    table->insert("ElementThree", std::string("THREE"));

    auto nested_table = cpptoml::make_table();
    nested_table->insert("ElementOne", 2L);
    nested_table->insert("ElementTwo", 3.0);
    nested_table->insert("ElementThree", std::string("FOUR"));

    // 父子表生成的方式
    table->insert("Nested", nested_table);

    root->insert("Table", table);

    // 生成数组
    auto int_array = cpptoml::make_array();
    int_array->push_back(1L);
    int_array->push_back(2L);
    int_array->push_back(3L);
    int_array->push_back(4L);
    int_array->push_back(5L);

    root->insert("IntegerArray", int_array);

    // 浮点型数组
    auto double_array = cpptoml::make_array();
    double_array->push_back(1.1);
    double_array->push_back(2.2);
    double_array->push_back(3.3);
    double_array->push_back(4.4);
    double_array->push_back(5.5);

    root->insert("DoubleArray", double_array);

    // 字符串类型的数组
    auto string_array = cpptoml::make_array();
    string_array->push_back(std::string("A"));
    string_array->push_back(std::string("B"));
    string_array->push_back(std::string("C"));
    string_array->push_back(std::string("D"));
    string_array->push_back(std::string("E"));

    root->insert("StringArray", string_array);

    // 表的数组
    auto table_array = cpptoml::make_table_array();
    table_array->push_back(table);
    table_array->push_back(table);
    table_array->push_back(table);

    root->insert("TableArray", table_array);

    // 数组的数组
    auto array_of_arrays = cpptoml::make_array();
    array_of_arrays->push_back(int_array);
    array_of_arrays->push_back(double_array);
    array_of_arrays->push_back(string_array);

    root->insert("ArrayOfArrays", array_of_arrays);

    std::cout << (*root);
    return 0;
}
