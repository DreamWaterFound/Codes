/**
 * @file MotionDetector_framesub.h
 * @author guoqing (1337841346@qq.com)
 * @brief 基于帧差法实现的运动目标检测
 * @version 0.1
 * @date 2019-01-07
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#ifndef __MOTION_DETECTOR_FRAMESUB_H__
#define __MOTION_DETECTOR_FRAMESUB_H__

#include "common.h"
#include "MotionDetector_DiffBase.h"

class MotionDetector_framesub : public MotionDetector_DiffBase
{
public:
    /**
     * @brief Construct a new MotionDetector_framesub object
     * 
     */
    MotionDetector_framesub();

    /**
     * @brief Destroy the MotionDetector_framesub object
     * 
     */
    ~MotionDetector_framesub();

public:

/**
 * @brief 计算差分图像
 * 
 * @param[in] frame     当前帧
 * @return cv::Mat      差分图像
 */
    virtual cv::Mat calcuDiffImg(cv::Mat frame);

public:
    //参数写入函数
    
    /**
     * @brief 重设所有参数、清空所有图像缓存
     * @details 清空图像缓存不是删除图片而是设置全黑的图片
     */
    virtual void resetDetector(void);

public:
    // 参数读出函数

    inline cv::Mat getImgGrayLastFrame(void) const
    {
        return mmGrayLastFrame;
    }     

protected:
    //私有成员变量


    ///上一帧图像
    //cv::Mat mmLastFrame;    
    ///上一帧的灰度图像
    cv::Mat mmGrayLastFrame;
    ///上一帧图像是否已经缓存
    bool mbIsLastFrameExist;


};

#endif //__MOTION_DETECTOR_FRAMESUB_H__