#include "MotionDetector_GMM.h"

using namespace std;
using namespace cv;

MotionDetector_GMM::MotionDetector_GMM()
{
    resetDetector();
}

MotionDetector_GMM::~MotionDetector_GMM()
{
    ;
}

//进行运动检测
cv::Mat MotionDetector_GMM::calcuDiffImg(cv::Mat frame)
{
    mpMOG2->apply(frame, mmDiff);
    return mmDiff;
}

void MotionDetector_GMM::resetDetector(void)
{
    //创建背景建模类
    mpMOG2=cv::createBackgroundSubtractorMOG2();
    resetDetector_base();
}