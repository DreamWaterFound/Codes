#ifndef __MOTION_DETECTOR_3_FRAMESUB__
#define __MOTION_DETECTOR_3_FRAMESUB__

#include "common.h"
#include "MotionDetector_framesub.h"

using namespace std;
using namespace cv;

class MotionDetector_3framesub :public MotionDetector_framesub
{
public:
    MotionDetector_3framesub();

    ~MotionDetector_3framesub();

public:

    cv::Mat calcuDiffImg(cv::Mat frame);

    virtual void resetDetector(void);
public:



private:
    ///上上帧图像
    cv::Mat mmGrayLastFrame2;
    ///上上帧 图像是否缓存的标志
    bool mbIsLastFrame2Exist;

};


#endif //__MOTION_DETECTOR_3_FRAMESUB__