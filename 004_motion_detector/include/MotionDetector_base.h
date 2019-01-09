/**
 * @file MotionDetector_base.h
 * @author guoqing (1337841346@qq.com)
 * @brief 运动目标检测算法类的基类声明
 * @version 0.1
 * @date 2019-01-07
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#ifndef __MOTION_DETECTOR_BASE_H__
#define __MOTION_DETECTOR_BASE_H__

#include "common.h"


/**
 * @brief 运动目标检测算法类的基类
 * @detials 不过目前只是用来存放一个帧的高度、宽度数据，有需要的时候再进行补充
 */
class MotionDetector_base
{
public:
    /**
     * @brief Construct a new MotionDetector_base object
     * 
     */
    MotionDetector_base();

    /**
     * @brief Destroy the MotionDetector_base object
     * 
     */
    ~MotionDetector_base();


public:
    //各种参数设置接口

    /**
     * @brief 设置帧的大小
     * 
     * @param[in] height    帧高度
     * @param[in] width     帧宽度
     * @return true         设置成功
     * @return false        设置失败，因为参数不合法
     */
    bool setFrameSize(int height,int width);

    /**
     * @brief 设置帧的大小
     * 
     * @param[in] size      size结构体
     * @return true         设置成功
     * @return false        设置失败，因为参数不合法
     */
    bool setFrameSize(cv::Size size);

public:
    //各种参数读取接口

    /**
     * @brief Get the Frame Size 
     * 
     * @return cv::Size 帧大小
     */
    inline cv::Size getFrameSize(void)
    {
        return mFrameSize;
    }

    /**
     * @brief Get the Frame Height 
     * 
     * @return int 帧高度
     */
    inline int getFrameHeight(void)
    {
        return mFrameSize.height;
    }

    /**
     * @brief Get the Frame Width 
     * 
     * @return int 帧宽度
     */
    inline int getFrameWidth(void)
    {
        return mFrameSize.width;
    }


    
protected:

    //好像参数的话，目前只是需要知道帧的大小就可以了
    cv::Size mFrameSize;

private:


};
#endif
