/**
 * @file DataReader.hpp
 * @author guoqing (1337841346@qq.com)
 * @brief 数据,数据集的读取器,基类实现
 * @version 0.1
 * @date 2019-02-16
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#ifndef __DATA_READER__
#define __DATA_READER__


#include <sstream>
#include <fstream>
#include <memory>

#include <opencv2/opencv.hpp>



namespace DataReader{

//using namespace std;
//using namespace cv;

//虚基类
class noncopyable
{
protected:
    noncopyable() = default;
    ~noncopyable() = default;

    noncopyable(const noncopyable&) = delete;
    noncopyable &operator=(const noncopyable&) = delete;
};

//数据读取器的基类
class DataReaderBase : public noncopyable
{
public:
    //typedef std::shared_ptr<DataReader> Ptr;

    DataReaderBase():mnNumber(0)
    {}
    ~DataReaderBase() {}
public:
    size_t mnNumber;    
};


}   //DataReader

#endif  //1__DATA_READER__