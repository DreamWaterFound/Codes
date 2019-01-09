#include "MotionDetector_backsub.h"

//================构造函数和析构函数====================
//构造函数
MotionDetector_backsub::MotionDetector_backsub()
{
    //复位各种参数
    resetDetector();
    //父类的暂时不考虑
}

//析构函数
MotionDetector_backsub::~MotionDetector_backsub()
{
    ;
}

//================主要功能实现====================
//获得差分图像
cv::Mat MotionDetector_backsub::calcuDiffImg(cv::Mat frame)
{
    //查看是否设置了背景模型图像
    if(!isBackgroundSet)
    {
        //如果没有设置
        isBackgroundSet=true;
        //不管返回值了
        setBackground(frame);
    }
    

    cvtColor(frame, mmGrayFrame, CV_BGR2GRAY);
    absdiff(mmGrayFrame, mmGrayBackground, mmDiff);
    return mmDiff;
}

//设置背景模型图像
bool MotionDetector_backsub::setBackground(cv::Mat background)
{
    //首先检查图像是否为空
    if(background.empty())
    {
        return false;
    }

    //设置
    mmBackground=background;
    //并且进行灰度化
    cv::cvtColor(mmBackground,mmGrayBackground,CV_BGR2GRAY);
    
    return true;
}

//================参数设置====================
void MotionDetector_backsub::resetDetector(void)
{
    
    mmBackground=cv::Mat(mFrameSize,CV_8UC3,Scalar(0,0,0));
    mmGrayBackground=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));
    isBackgroundSet=false;

    resetDetector_base();
}
