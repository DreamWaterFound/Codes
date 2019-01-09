/**
 * @file MotionDetector_GMM2.h
 * @author guoqing (1337841346@qq.com)
 * @brief 通过自己的方式来实现的GMM
 * @details 也是用这种方式来实现前景检测
 * @version 0.1
 * @date 2019-01-08
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#ifndef __MOTION_DETETCTOR_GMM_2_H__
#define __MOTION_DETETCTOR_GMM_2_H__



#include "common.h"
#include "MotionDetector_DiffBase.h"

using namespace std;
using namespace cv;


/** @brief GMM模型的最多数目,这个数值是定死的,不能够在运行时改变 */
#define GMM_MAX_COMPONT 2
//学习速率
#define DEFAULT_GMM_LEARN_ALPHA 0.005    //该学习率越大的话，学习速度太快，效果不好
//阈值
#define DEFALUT_GMM_THRESHOD_SUMW 0.7    //如果取值太大了的话，则更多部分都被检测出来了
//学习帧数
#define DEFAULT_END_FRAME 20

         
//几个用来加速操作的宏
#define W(index,x,y) (mmWeight[index].at<float>(x,y))
#define U(index,x,y) (mmU[index].at<unsigned char>(x,y))
#define Sigma(index,x,y) (mmSigma[index].at<float>(x,y))
#define IMG(x,y) (img.at<unsigned char>(x,y))


class MotionDetector_GMM2 : public MotionDetector_DiffBase
{
public:
    /**
     * @brief Construct a new MotionDetector_GMM2 object
     * 
     */
    MotionDetector_GMM2();

    /**
     * @brief Destroy the MotionDetector_GMM2 object
     * 
     */
    ~MotionDetector_GMM2();

public:

    /**
     * @brief 使用高斯混合模型的方式来的到差分图像
     * 
     * @param[in] frame     当前时刻的帧
     * @return cv::Mat      计算得到的差分图像
     */
    cv::Mat calcuDiffImg(cv::Mat frame);

    /**
     * @brief 重设本基类所有参数、清空所有图像缓存
     * @details 清空图像缓存不是删除图片而是设置全黑的图片
     */
    void resetDetector(void);

private:
    //私有函数

    /**
     * @brief 初始化高斯模型的所有参数
     * 
     * @param img 第一帧图像
     */
    void gmmInit(cv::Mat img);

    /**
     * @brief 处理第一帧的函数
     * 
     * @param img 第一帧图像
     */
    void gmmDealFirstFrame(cv::Mat img);

    /**
     * @brief 对GMM模型进行训练的函数
     * 
     * @param img 当前帧图像
     */
    void gmmTrain(cv::Mat img);

    /**
     * @brief 根据当前帧确定每个像素最适合使用的高斯模型是多少个
     * 
     * @param img 当前帧图像
     */
    void gmmCalFitNum(cv::Mat img);

    /**
     * @brief 使用GMM来进行前景检测
     * 
     * @param img 当前帧图像
     */
    void gmmTest(cv::Mat img);

public:
    //参数设置函数,现在先不写
    //TODO 

    
private:
    

private:

    /**
     * @name 高斯模型参数
     * @{
     */

    //私有变量
    /** 每个像素的每个高斯模型的权值 */
    cv::Mat mmWeight[GMM_MAX_COMPONT];
    /** 每个像素的每个高斯模型的均值 */
    cv::Mat mmU[GMM_MAX_COMPONT];
    /** 每个像素的每个高斯模型的 协方差矩阵 */
    cv::Mat mmSigma[GMM_MAX_COMPONT];

    /** @} */

    /**
     * @brief 其他
     * @{
     */

    /** 表示某个像素最实用的高斯模型的数目 */
    cv::Mat mmFitNum;
    /** 学习率 */
    float mfLearnRate;
    /** 判断 fitNum 时的权重累加和阈值 */
    float mfThreshod;
    /** 学习帧数 */
    int mnLearnFrameNumber;

    /** 当前视频帧数 */
    int  mnFrameCnt;


    /** @} */

    /** 最后生成的掩摸图像 */
    cv::Mat mmGMMMask;



};

#endif //__MOTION_DETETCTOR_GMM_2_H__