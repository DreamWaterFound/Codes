/**
 * @file MotionDetector_GMM.h
 * @author guoqing (1337841346@qq.com)
 * @brief GMM模型实现
 * @version 0.1
 * @date 2019-01-08
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "common.h"
#include "MotionDetector_DiffBase.h"

using namespace std;
using namespace cv;


class MotionDetector_GMM:public MotionDetector_DiffBase
{
public:
    /**
     * @brief Construct a new MotionDetector_GMM object
     * 
     */
    MotionDetector_GMM();

    /**
     * @brief Destroy the MotionDetector_GMM object
     * 
     */
    ~MotionDetector_GMM();

public:
    
    /**
     * @brief 计算差分图像
     * 
     * @param[in] frame     当前帧
     * @return cv::Mat  差分图像
     */
    cv::Mat calcuDiffImg(cv::Mat frame);

    /**
     * @brief 重设所有参数、清空所有图像缓存
     * @details 清空图像缓存不是删除图片而是设置全黑的图片
     */
    void resetDetector(void);


private:

    Ptr<BackgroundSubtractor> mpMOG2;        ///<背景建模类？

};