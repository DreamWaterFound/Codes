/**
 * @file my_example.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 这个事情还是得需要我自己去尝试啊
 * @version 0.1
 * @date 2019-07-03
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include <iostream>
#include <string>

#include "tools/cpptoml.h"

using namespace std;

void parse_arguments(string file_path);


int main(int argc, char* argv[])
{
    cout<<"My Test for cpptoml."<<endl;
    cout<<"Compled at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    if(argc!=2)
    {
        cout<<"Usage: "<<argv[0]<<" toml_file_path"<<endl;
        return 0;
    }

    try
    {
        parse_arguments(argv[1]);
    }
    catch (const cpptoml::parse_exception& ex)
    {
        std::cerr << "Parsing failed: " << ex.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Something horrible happened!" << std::endl;
        // return as if there was success so that toml-test will complain
        return 0;
    }
    

    return 0;
}

void parse_arguments(string file_path)
{
    auto config = cpptoml::parse_file(file_path);

    cout<<"============================================"<<endl;
    // 整数
    int64_t my_int=static_cast<int64_t>(*config->get_as<int64_t>("my_int"));
    cout<<"my_int="<<my_int<<endl;

    // 浮点数
    float my_float=static_cast<float>(*config->get_as<double>("my_float"));
    cout<<"my_float="<<my_float<<endl;

    // 字符串
    string my_string=*config->get_as<string>("my_string");
    cout<<"my_string="<<my_string<<endl;

    // 日期， 但是考虑到一般使用不到，所以这里就不进行相关的实验了

    // 列表 注意访问列表的时候其中的调用的函数已经改变了
    string name=*config->get_qualified_as<string>("profile.name");
    cout<<"name="<<name<<endl;
    int64_t id=static_cast<int64_t>(*config->get_qualified_as<int64_t>("profile.id"));
    cout<<"id="<<id<<endl;

    // 嵌套列表的使用 -- 一种方法是直接使用点的方式
    int chinese_scorce=static_cast<int>(*config->get_qualified_as<int>("profile.scorce.chinese"));
    cout<<"chinese_scorce="<<chinese_scorce<<endl;
    int math_scorce=static_cast<int>(*config->get_qualified_as<int>("profile.scorce.math"));
    cout<<"math_scorce="<<math_scorce<<endl;
    int english_scorce=static_cast<int>(*config->get_qualified_as<int>("profile.scorce.english"));
    cout<<"english_scorce="<<english_scorce<<endl;
    
    // 嵌套列表的使用 -- 另外一种方法是先获取列表对象，然后逐个获取元素
    auto profile_table=config->get_table("profile");        // 经过测试发现还得是这样写,这里就不能够再使用点的方式了
    auto grade_table=profile_table->get_table("grade");
    string pe_grade=*grade_table->get_as<string>("pe");
    cout<<"pe_grade="<<pe_grade<<endl;

    // 列表
    auto arr_table=config->get_table("arr");
    vector<int64_t> arr_same=static_cast<vector<int64_t> >(*arr_table->get_array_of<int64_t>("arr_same"));
    cout<<"arr_same:"<<endl;
    for(const int64_t val:arr_same)
    {
        cout<<"\t"<<val<<endl;
    }

    // 嵌套列表
    auto arr_mixed=arr_table->get_array_of<cpptoml::array>("arr_mixed");
    vector<int64_t> ints=static_cast<vector<int64_t> >(*(*arr_mixed)[0]->get_array_of<int64_t>());
    vector<string> strings=static_cast<vector<string> >(*(*arr_mixed)[1]->get_array_of<string>());
    vector<double> doubles=static_cast<vector<double> >(*(*arr_mixed)[2]->get_array_of<double>());
    cout<<"arr_mixed:"<<endl<<"\tints"<<endl;
    for(const int64_t val:ints)    {        cout<<"\t\t"<<val<<endl;    }
    cout<<"\tstrings"<<endl;
    for(const string val:strings)  {        cout<<"\t\t"<<val<<endl;    }
    cout<<"\tdoubles"<<endl;
    for(const double val:doubles)  {        cout<<"\t\t"<<val<<endl;    }

    // 表本身的列表就不进行测试了

    cout<<"============================================"<<endl;
}