#include "MotionDetector_framesub.h"

//================构造函数和析构函数====================
//构造函数
MotionDetector_framesub::MotionDetector_framesub():
    mbIsLastFrameExist(false)
{
    //复位各种参数
    resetDetector();
    //父类的暂时不考虑
}

//析构函数
MotionDetector_framesub::~MotionDetector_framesub()
{
    ;
}

//================主要功能实现====================
//计算差分图像
cv::Mat MotionDetector_framesub::calcuDiffImg(cv::Mat frame)
{
    //检查一下上一帧是否存储
    if(!mbIsLastFrameExist)
    {
        mbIsLastFrameExist=true;
        //灰度化
        cv::cvtColor(frame,mmGrayFrame,CV_BGR2GRAY);
        //本帧也是上一帧的灰度图像
        mmGrayLastFrame=mmGrayFrame.clone();
    }
    else
    {
        mmGrayLastFrame=mmGrayFrame.clone();
        //灰度化
        cv::cvtColor(frame,mmGrayFrame,CV_BGR2GRAY);
    }

    //作差
    cv::absdiff(mmGrayLastFrame,mmGrayFrame,mmDiff);

    return mmDiff;
}

//================参数设置====================
void MotionDetector_framesub::resetDetector(void)
{
    mmGrayLastFrame=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));
    mbIsLastFrameExist=false;
    resetDetector_base();
}
