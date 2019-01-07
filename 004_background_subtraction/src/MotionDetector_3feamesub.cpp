#include "MotionDetector_3framesub.h"


MotionDetector_3framesub::MotionDetector_3framesub():
    mbIsLastFrame2Exist(false)
{
    resetDetector();
}

MotionDetector_3framesub::~MotionDetector_3framesub()
{
    ;
}

cv::Mat MotionDetector_3framesub::calcuDiffImg(cv::Mat frame)
{
    //检查一下上一帧是否存储
    if(!mbIsLastFrameExist)
    {
        mbIsLastFrameExist=true;
        //灰度化
        cv::cvtColor(frame,mmGrayFrame,CV_BGR2GRAY);
        //本帧也是上一帧的灰度图像
        mmGrayLastFrame=mmGrayFrame.clone();
        //同时还是上上一帧的图像
        mmGrayLastFrame2=mmGrayFrame.clone();
    }
    else
    {
        //上上一帧图像
        mmGrayLastFrame2=mmGrayLastFrame.clone();
        //上一帧图像
        mmGrayLastFrame=mmGrayFrame.clone();
        //灰度化
        cv::cvtColor(frame,mmGrayFrame,CV_BGR2GRAY);
    }

    //作差
    cv::absdiff(mmGrayLastFrame2,mmGrayFrame,mmDiff);

    return mmDiff;
}

void MotionDetector_3framesub::resetDetector(void)
{
    //这里要把父类的初初始化也给做了
    mmGrayLastFrame=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));
    mbIsLastFrameExist=false;

    //自己的
    mmGrayLastFrame2=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));
    mbIsLastFrame2Exist=false;
    //爷爷类的。。。
    resetDetector_base();
}