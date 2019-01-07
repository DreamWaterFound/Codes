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

#include "common.h"
#include "MotionDetector_base.h"

/** @brief 最大图像灰度强度 */
#define MAX_INTENSITY 255

#define DEFAULT_THRESHOLD 50
#define DEFAULT_EROPE_KERNEL_HEIGHT 3
#define DEFAULT_EROPE_KERNEL_WIDTH 3
#define DEFAULT_DILATE_KERNEL_HEIGHT 15
#define DEFAULT_DILATE_KERNEL_WIDTH 15
#define DEFAULT_COLOR_RED (cv::Scalar(0,0,255))
#define DEFAULT_COLOR_GREEN (cv::Scalar(0,255,0))
#define DEFAULT_COMTOUR_LINE_WIDTH 2
#define DEFAULT_RECTANGLE_LINE_WIDTH 2





/**
 * @brief 实现背景减法来进行运动目标检测的类
 * @detials 继承自  MotionDetector_base 类。
 */
class MotionDetector_backsub : MotionDetector_base 
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
     * @brief 对当前帧进行运动目标检测
     * 
     * @param[in] frame 当前帧
     * @return cv::Mat  运动目标检测结果
     * @note 注意这里并不会进行图像尺寸的检查，请确保和背景模型一致
     */
    cv::Mat motionDetect(cv::Mat frame);

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

    /**
     * @brief 设置腐蚀核的大小并生成腐蚀核，锚点位置默认
     * 
     * @param[in] heigh 高度
     * @param[in] width 宽度
     * @return true     设置成功
     * @return false    设置失败，一般是参数不合格
     */
    bool setEropeKernelSize(int heigh, int width);

    /**
     * @brief 设置腐蚀核的大小并生成腐蚀核，锚点位置默认
     * @details 这个函数默认腐蚀核是正方形的
     * @param[in] size      大小
     * @return true     设置成功
     * @return false    设置失败，参数范围不合格 
     */
    bool setEropeKernelSize(cv::Size size);

    /**
     * @brief 设置膨胀核大小，并且生成部件
     * 
     * @param[in] heigh     高度
     * @param[in] width     宽度
     * @return true         设置成功
     * @return false        设置失败，参数不合法
     */
    bool setDilateKernelSize(int height,int width);

    /**
     * @brief 设置膨胀核大小，并且生成部件
     * 
     * @param[in] size      大小
     * @return true         设置成功
     * @return false        设置失败，参数不合法
     */
    bool setDilateKernelSize(cv::Size size);
    
    /**
     * @brief 设置二值化时的阈值
     * 
     * @param[in] thr   阈值
     * @return true     设置成功
     * @return false    设置失败，参数不合法(0~255)
     */
    bool setBinaryThreshold(int thr);

    /**
     * @brief 设置轮廓颜色
     * 
     * @param[in] r     红色分量
     * @param[in] g     绿色分量
     * @param[in] b     蓝色分量
     * @return true     设置成功
     * @return false    设置失败，参数不在范围内(0~255)
     */
    bool setContourColor(int r,int g,int b);

    /**
     * @brief 设置轮廓颜色
     * 
     * @param[in] color 颜色
     * @return true     设置成功
     * @return false    设置失败，参数不在范围内(每个分量0~255)
     */
    bool setContourColor(cv::Scalar color);

    /**
     * @brief 设置轮廓外接矩形颜色
     * 
     * @param[in] r     红色分量
     * @param[in] g     绿色分量
     * @param[in] b     蓝色分量
     * @return true     设置成功
     * @return false    设置失败，参数不在范围内(0~255)
     */
    bool setRectangleColor(int r,int g,int b);

     /**
     * @brief 设置轮廓外接矩形颜色
     * 
     * @param[in] color 颜色
     * @return true     设置成功
     * @return false    设置失败，参数不在范围内(每个分量0~255)
     */
    bool setRectangleColor(cv::Scalar color);

    /**
     * @brief Set the Contour Line Width 
     * 
     * @param[in] witdh     宽度
     * @return true         设置成功
     * @return false        设置失败，参数不在范围内(>0)
     */
    bool setContourLineWidth(int width);

    /**
     * @brief Set the Rectangle Line Width
     * 
     * @param[in] witdh     宽度
     * @return true         设置成功
     * @return false        设置失败，参数不在范围内(>0)
     */
    bool setRectangleLineWidth(int width);
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

    inline cv::Mat getGrayFrame(void) const
    {
        return mmGrayFrame;
    }

    inline cv::Mat getImgDiff(void) const
    {
        return mmDiff;
    }

    inline cv::Mat getImgDiffThresh(void) const
    {
        return mmDiffThresh;
    }

    inline cv::Mat getImgErode(void) const
    {
        return mmErode;
    }

    inline cv::Mat getImgDilate(void) const
    {
        return mmDilate;
    }

    inline cv::Mat getImgContours(void) const
    {
        return mmContours;
    }

    inline cv::Mat getImgResult(void) const
    {
        return mmResult;
    }



private:

    /**
     * @brief 检查一个cv::Scalar类型的数据是否能够表示一个颜色
     * 
     * @param[in] color     数据
     * @return true         是
     * @return false        否
     */
    bool isColor(cv::Scalar color);

private:
    //私有成员变量

    /** 
     * @name 图像类
     * @{
     */

    ///背景模型图像
    cv::Mat mmBackground;    
    ///灰度的背景模型图像
    cv::Mat mmGrayBackground;
    ///灰度的当前帧图像
    cv::Mat mmGrayFrame;
    ///差分图像
    cv::Mat mmDiff;
    ///阈值化后的差分图像
    cv::Mat mmDiffThresh;
    ///腐蚀后的图像
    cv::Mat mmErode;
    ///膨胀图像
    cv::Mat mmDilate;
    ///绘制有轮廓的图像
    cv::Mat mmContours;
    ///叠加有最终结果的图像
    cv::Mat mmResult;

    /** @} */

    /**
     * @name 参数类
     * @{
     */

    ///阈值化时的阈值
    int mnThreshold;
    ///腐蚀核大小
    cv::Size mErodeKernelSize;
    ///膨胀核大小
    cv::Size mDilateKernelSize;
    ///轮廓颜色
    cv::Scalar mContourColor;
    ///边框颜色
    cv::Scalar mRectColor;
    ///轮廓线条宽度
    int mnContourLineWidth;
    ///外接矩形线条宽度
    int mnRectangleLineWidth;

    /** @} */



    /**
     * @name 其他
     * @{
     */

    ///腐蚀用的结构元素
    cv::Mat mmKernelErode;
    ///膨胀用的结构元素
    cv::Mat mmKernelDilate;
    ///轮廓点集
    vector<vector<cv::Point> > *mpvvContours;
    ///外接矩形
    vector<Rect> *mpvRectangles;

    /** @} */

};