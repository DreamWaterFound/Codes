/**
 * @file example_my.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 我自己的测试
 * @version 0.1
 * @date 2019-07-02
 * @copyright Copyright (c) 2019
 */

#include <iostream>

#include "tools/cxxopts.hpp"

using namespace std;

int main(int argc, char* argv[])
{
  // 由于程序会抛出异常，所以在这个过程中要有这样子的一个 try..catch 对
  try
  {
    cxxopts::Options options(argv[0],
        "my Example progeam.");

    options
      .positional_help("[optional args]")   
      .show_positional_help();              
    
    options.add_options()
      ("d,datasets","Datasets path",cxxopts::value<std::string>())
      ("a,associate","associate file",cxxopts::value<std::string>())
      ;

    // 
    options.parse_positional({"associate","datasets"});

    auto result = options.parse(argc, argv);

    if(result.count("help"))
    {
      cout<<options.help({""})<<endl;
      exit(0);
    }


    if(result.count("datasets"))
    {
      cout<<"Datasets: "<<result["datasets"].as<string>()<<endl;
    }

    if(result.count("associate"))
    {
      cout<<"Associate: "<<result["associate"].as<string>()<<endl;
    }

    std::cout << "Arguments remain = " << argc << std::endl;

    cout<<"remain:"<<argv[argc-1]<<endl;
    
    return 0;

  } catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }

  return 0;
}
