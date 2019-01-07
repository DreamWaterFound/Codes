/**
 * @file MotionDetector.h
 * @author guoqing (1337841346@qq.com)
 * @brief 运动检测算法的类声明文件
 * @version 0.1
 * @date 2019-01-07
 * 
 * @copyright Copyright (c) 2019
 * 
 */


#include "common.h"

using namespace std;

/**
 * @brief 运动物体检测器。
 * @detials 提供一系列的运动检测方法。为了方便，现在暂时将这个类定义为静态类。
 */
class MotionDetector
{
public:
    /**
     * @brief 使用背景减法来进行物体检测
     * 
     * @param[in] background    背景图像
     * @param[in] frame         当前帧
     * @return cv::Mat          含有轮廓的外接矩形的检测结果结果
     */
    static cv::Mat back_sub(cv::Mat background, cv::Mat frame);

    
};