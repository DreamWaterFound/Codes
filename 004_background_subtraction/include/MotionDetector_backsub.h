/**
 * @file MotionDetector_backsub.h
 * @author guoqing (1337841346@qq.com)
 * @brief 基于背景减法实现的运动目标检测
 * @version 0.1
 * @date 2019-01-07
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#ifndef __MOTION_DETECTOR_BACKSUB_H__
#define __MOTION_DETECTOR_BACKSUB_H__

#include "common.h"
#include "MotionDetector_DiffBase.h"

/**
 * @brief 实现背景减法来进行运动目标检测的类
 * @detials 继承自  MotionDetector_DiffBase 类。
 */
class MotionDetector_backsub : public MotionDetector_DiffBase 
{
public:
    /**
     * @brief Construct a new MotionDetector_backsub object
     * 
     */
    MotionDetector_backsub();

    /**
     * @brief Destroy the MotionDetector_backsub object
     * 
     */
    ~MotionDetector_backsub();

public:
    
    /**
     * @brief 计算和背景模型的差分图像
     * 
     * @param[in] frame     当前帧
     * @return cv::Mat  差分图像
     */
    cv::Mat calcuDiffImg(cv::Mat frame);

    /**
     * @brief Set the Background 
     * 
     * @param[in] background 背景图片
     * @return true     设置成功
     * @return false    设置失败，一般是因为图片为空
     */
    bool setBackground(cv::Mat background);

public:
    //参数写入函数
    
    /**
     * @brief 重设所有参数、清空所有图像缓存
     * @details 清空图像缓存不是删除图片而是设置全黑的图片
     */
    void resetDetector(void);

public:
    // 参数读出函数
    
    inline cv::Mat getImgBackGround(void) const
    {
        return mmBackground;
    }

    inline cv::Mat getImgGrayBackground(void) const
    {
        return mmGrayBackground;
    }

private:
    //私有成员变量

    ///背景模型图像
    cv::Mat mmBackground;    
    ///灰度的背景模型图像
    cv::Mat mmGrayBackground;

};
#endif