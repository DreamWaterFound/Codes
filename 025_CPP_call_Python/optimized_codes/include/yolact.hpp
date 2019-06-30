/**
 * @file yolact.hpp
 * @author guoqing (1337841346@qq.com)
 * @brief 提供对YOLACT网络的接口操作；底层操作还是通过调用Python函数实现的
 * @version 0.1
 * @date 2019-06-27
 * 
 * @copyright Copyright (c) 2019
 * 
 */


#ifndef __YOLACT_HPP__
#define __YOLACT_HPP__


// ==============================  头文件  =============================

// C++基础
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

// Python 支持
#include <Python.h>
#include <numpy/arrayobject.h>

// OpenCV支持
#include <opencv2/opencv.hpp>

// ==============================  宏定义  =============================

// 避免出现警告，但是好像是就算加上去了也没有什么卵用
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

// #ifndef NPY_1_7_API_VERSION
// #define 
// #endif

// 不要在头文件中引用任何名字空间

/** @brief Yolact */
namespace YOLACT{

/** @brief 定义了对YOLACT网络的各种操作 */
class YOLACT
{
public:
    // 构造函数
    YOLACT( std::string pyEnvPkgsPath,
            std::string pyMoudlePathAndName,
            std::string initPyFunctionName,
            std::string evalPyfunctionName,
            std::string trainedModelPath,
            float       scoreThreshold     =0.0,
            int         topK               =5,
            bool        detect             =false,
            bool        crossClassNms      =true,
            bool        fastNms            =true,
            bool        displayMasks       =true,
            bool        displayBBoxes      =true,
            bool        displayText        =true,
            bool        displayScores      =true,
            bool        displayLincomb     =false,
            bool        maskProtoDebug     =false);
        
    // 析构函数
    ~YOLACT();

public:

    // 查看是否初始化成功
    inline bool isInitializedResult(void) const
    {   return mbIsPythonInitializedOK & mbIsYOLACTInitializedOK;   }
    // 查看错误提示
    inline const std::string& getErrorDescriptionString(void) const
    {   return mstrErrDescription;    }

    // 获取网络可以识别的类别数目
    inline size_t getCLassNum(void) const
    {   return mnClassNum;    }

    // 获取网络可以识别的类别名称
    // 为了避免破坏封装性，这里要求必须是常值引用
    inline const std::vector<std::string>& getClassNames(void) const 
    {   return mvstrClassNames;    }

    /**
     * @brief 评估图像 - 直接拿到类名
     * @param[in]  srcImage         要进行处理的图像
     * @param[out] resImage         渲染结果后的图像
     * @param[out] vstrClassName    类名
     * @param[out] vdScores         评分
     * @param[out] vpairBBoxes      bounding boxes
     * @param[out] vimgMasks        masks
     * @return true                 处理ok
     * @return false                处理失败
     */
    bool EvalImage(const cv::Mat& srcImage,
                   cv::Mat& resImage,
                   std::vector<std::string>& vstrClassName,
                   std::vector<float>& vdScores,
                   std::vector<std::pair<cv::Point2i,cv::Point2i> >& vpairBBoxes,
                   std::vector<cv::Mat>& vimgMasks);

    /**
     * @brief 评估图像 - 直接拿到类id
     * @param[in]  srcImage     要进行处理的图像
     * @param[out] resImage     渲染结果后的图像
     * @param[out] vstrClassId  类名
     * @param[out] vdScores     评分
     * @param[out] vpairBBoxes  bounding boxes
     * @param[out] vimgMasks    masks
     * @return true             处理ok
     * @return false            处理失败
     */
    bool EvalImage(const cv::Mat& srcImage,
                   cv::Mat& resImage,
                   std::vector<size_t>& vstrClassId,
                   std::vector<float>& vdScores,
                   std::vector<std::pair<cv::Point2i,cv::Point2i> >& vpairBBoxes,
                   std::vector<cv::Mat>& vimgMasks);

private:

    bool ImportNumPySupport(void) const
    {
        // 这是一个宏，其中包含了返回的语句
        import_array();
        return true;
    }

    bool Image2Numpy(const cv::Mat& srcImage,
                PyObject*& pPyArray);

private:

    // Python 文件和模块相关
    std::string     mstrPyMoudlePath;
    std::string     mstrPyMoudleName;
    std::string     mstrEvalPyfunctionName;
    PyObject*       mpPyEvalModule;
    PyObject*       mpPyEvalFunc;
    
    // 保存类型
    size_t mnClassNum;
    std::vector<std::string> mvstrClassNames;

    // 用于在图像转换时的数组指针
    unsigned char  *mpb8ImgTmpArray;

    // 用于指示错误的
    bool            mbIsYOLACTInitializedOK;
    bool            mbIsPythonInitializedOK;
    std::string     mstrErrDescription;

    // 一些中间的文件
    PyObject *mpPyArgList;
    PyObject *mpPyRetValue;

    


};      // class YOLACT

}       // namespace YOLACT

#endif  // __YOLACT_HPP__
