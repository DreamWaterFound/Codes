/**
 * @file MotionDetector_base.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 运动目标检测算法类的基类实现
 * @version 0.1
 * @date 2019-01-07
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#include "MotionDetector_base.h"

using namespace std;
using namespace cv;     //虽然自己写的时候习惯加上前缀，但是这里还是这样写一下吧

//构造函数
MotionDetector_base::MotionDetector_base():
    mFrameSize(cv::Size(0,0))
{
    ;
}

//析构函数
MotionDetector_base::~MotionDetector_base()
{
    ;
}

//设置帧的大小
bool MotionDetector_base::setFrameSize(int height,int width)
{
    return setFrameSize(cv::Size(height,width));
}

//设置帧的大小
bool MotionDetector_base::setFrameSize(cv::Size size)
{
    //检查参数
    if(size.height<1 || size.width<1)
    {
        return false;
    }
    else
    {
        mFrameSize=size;
        return true;
    }
}


