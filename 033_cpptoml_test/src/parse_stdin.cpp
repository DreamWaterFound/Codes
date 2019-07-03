/**
 * @file parse_stdin.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 从命令行解析参数的demo  //? 
 * @version 0.1
 * @date 2019-07-03
 * 
 * @copyright Copyright (c) 2019
 * 
 */


#include "tools/cpptoml.h"

#include <iostream>
#include <limits>

/**
 * A visitor for toml objects that writes to an output stream in the JSON
 * format that the toml-test suite expects.
 */
/** @ brief 目测是一个转换用的? */
class toml_test_writer
{
  public:
    /**
     * @brief 构造函数
     * @param[in] s 一个流 //? 干嘛用的?
     */
    toml_test_writer(std::ostream& s) : stream_(s)
    {
        // nothing
    }

    void visit(const cpptoml::value<std::string>& v)
    {
        stream_ << "{\"type\":\"string\",\"value\":\""
                << cpptoml::toml_writer::escape_string(v.get()) << "\"}";
    }

    void visit(const cpptoml::value<int64_t>& v)
    {
        stream_ << "{\"type\":\"integer\",\"value\":\"" << v.get() << "\"}";
    }

    void visit(const cpptoml::value<double>& v)
    {
        stream_ << "{\"type\":\"float\",\"value\":\"" << v.get() << "\"}";
    }

    void visit(const cpptoml::value<cpptoml::local_date>& v)
    {
        stream_ << "{\"type\":\"local_date\",\"value\":\"" << v.get() << "\"}";
    }

    void visit(const cpptoml::value<cpptoml::local_time>& v)
    {
        stream_ << "{\"type\":\"local_time\",\"value\":\"" << v.get() << "\"}";
    }

    void visit(const cpptoml::value<cpptoml::local_datetime>& v)
    {
        stream_ << "{\"type\":\"local_datetime\",\"value\":\"" << v.get()
                << "\"}";
    }

    void visit(const cpptoml::value<cpptoml::offset_datetime>& v)
    {
        stream_ << "{\"type\":\"datetime\",\"value\":\"" << v.get() << "\"}";
    }

    void visit(const cpptoml::value<bool>& v)
    {
        stream_ << "{\"type\":\"bool\",\"value\":\"" << v << "\"}";
    }

    void visit(const cpptoml::array& arr)
    {
        stream_ << "{\"type\":\"array\",\"value\":[";
        auto it = arr.get().begin();
        while (it != arr.get().end())
        {
            (*it)->accept(*this);
            if (++it != arr.get().end())
                stream_ << ", ";
        }
        stream_ << "]}";
    }

    void visit(const cpptoml::table_array& tarr)
    {
        stream_ << "[";
        auto arr = tarr.get();
        auto ait = arr.begin();
        while (ait != arr.end())
        {
            (*ait)->accept(*this);
            if (++ait != arr.end())
                stream_ << ", ";
        }
        stream_ << "]";
    }

    void visit(const cpptoml::table& t)
    {
        stream_ << "{";
        auto it = t.begin();
        while (it != t.end())
        {
            stream_ << '"' << cpptoml::toml_writer::escape_string(it->first)
                    << "\":";
            it->second->accept(*this);
            if (++it != t.end())
                stream_ << ", ";
        }
        stream_ << "}";
    }

  private:
    // ? 不知道是用来做什么用的流
    std::ostream& stream_;
};


/**
 * @brief 主函数
 * @return int state
 */
int main()
{
    // 控制cout的输出的有效数字,这里对应的是double型数据的最大有效数字
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    // 构造对象
    cpptoml::parser p{std::cin};
    try
    {
        std::shared_ptr<cpptoml::table> g = p.parse();
        toml_test_writer writer{std::cout};
        g->accept(writer);
        std::cout << std::endl;
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
