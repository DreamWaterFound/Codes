/*

Copyright (c) 2014 Jarryd Beck

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include <iostream>

#include "tools/cxxopts.hpp"

int main(int argc, char* argv[])
{
  // 由于程序会抛出异常，所以在这个过程中要有这样子的一个 try..catch 对
  try
  {
    // 构造对象
    cxxopts::Options options(argv[0],                               // 当前应用程序的名字
                             " - example command line options");    // 对应用程序的描述
    // 这种连续操作的方式我喜欢
    options
      .positional_help("[optional args]")   // 自定义当执行 --help 的时候在 optional args 处应该显示的信息
      .show_positional_help();              // 设置当执行 --help 的时候显示 optional args 的帮助信息

    // 初始值
    bool apple = false;

    // 给当前应用程序添加帮助信息 -- 也是链式操作
    options.add_options()
      ("a,apple", "an apple", cxxopts::value<bool>(apple))                          // 这个选项其实还和变量apple进行了一个"绑定"
      ("b,bob", "Bob")                                                              // 也可以不和变量绑定,直接这样写
      ("t,true", "True", cxxopts::value<bool>()->default_value("true"))             // bool型的变量缺省值
      ("f,file", "File", cxxopts::value<std::vector<std::string>>(), "FILE")        // 无缺省值;但是不会从第一个字符串数组类型的开始
      ("i,input", "Input", cxxopts::value<std::string>())                           // 无缺省值的参数.默认不指定选项或参数的话,将会先分配至这个(因为这个是第一个没有缺省值的)
      ("o,output", "Output file", cxxopts::value<std::string>()                     // 同时指定缺省值和隐含值
          ->default_value("a.out")->implicit_value("b.def"), "BIN")
      ("positional",                                                                // 剩下的就都会被归类到这个参数里面
        "Positional arguments: these are the arguments that are entered "
        "without an option", cxxopts::value<std::vector<std::string>>())
      ("long-description",
        "thisisareallylongwordthattakesupthewholelineandcannotbebrokenataspace")
      ("help", "Print help")
      ("int", "An integer", cxxopts::value<int>(), "N")
      ("float", "A floating point number", cxxopts::value<float>())
      ("option_that_is_too_long_for_the_help", "A very long option")
    #ifdef CXXOPTS_USE_UNICODE
      ("unicode", u8"A help option with non-ascii: à. Here the size of the"
        " string should be correct")
    #endif
    ;

    // 目测是有些函数可以按功能进行分组,这里就是进行分组用的;但是其实分组对参数的解析没有任何影响,不过是在输出帮助信息的时候会有一定的帮助
    options.add_options("Group")
      ("c,compile", "compile")
      ("d,drop", "drop", cxxopts::value<std::vector<std::string>>());

    // 对于那些没有指定-或者--的,那么就是按照位置解析,顺序如下
    options.parse_positional({"input", "output", "positional"});

    // 得到解析之后的结果
    auto result = options.parse(argc, argv);

    // 如果命令行中给出了这个命令,那么这个命令的计数就不是0,就可以进入到这个大括号中
    if (result.count("help"))
    {
      // 输出这两个组的参数的帮助信息. "" 表示默认的组
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(0);
    }

    // 如果已经命令行参数中已经给出过了这个参数
    if (apple)
    {
      // 我发现这里 result.count("a") 和 result.count("apple") 得到的命令行选项计数是完全相同的
        std::cout << "Saw option ‘a’ " << result.count("a") << " times " <<
        std::endl;
    }

    // 同样的,不过这个只是显示出命令b被给定的次数
    if (result.count("b"))
    {
      std::cout << "Saw option ‘b’" << std::endl;
    }

    // 这种多个的情况命令行要写 -f file1 -f file2 这样子才能够确保能够被程序正确解析
    if (result.count("f"))
    {
      // 注意这里拿到的是一个字符串组; 同时注意下标的访问方式
      auto& ff = result["f"].as<std::vector<std::string>>();
      std::cout << "Files" << std::endl;
      // 挨个输出文件名称
      for (const auto& f : ff)
      {
        std::cout << f << std::endl;
      }
    }

    // 输出对应的 参数
    if (result.count("input"))
    {
      std::cout << "Input = " << result["input"].as<std::string>()
        << std::endl;
    }

    if (result.count("output"))
    {
      std::cout << "Output = " << result["output"].as<std::string>()
        << std::endl;
    }

    if (result.count("positional"))
    {
      std::cout << "Positional = {";
      auto& v = result["positional"].as<std::vector<std::string>>();
      for (const auto& s : v) {
        std::cout << s << ", ";
      }
      std::cout << "}" << std::endl;
    }

    if (result.count("int"))
    {
      std::cout << "int = " << result["int"].as<int>() << std::endl;
    }

    if (result.count("float"))
    {
      std::cout << "float = " << result["float"].as<float>() << std::endl;
    }

    // 这里还剩一个也是正常的,因为剩的这一个就是原来的argv[0]
    std::cout << "Arguments remain = " << argc << std::endl;

  } catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }

  return 0;
}
