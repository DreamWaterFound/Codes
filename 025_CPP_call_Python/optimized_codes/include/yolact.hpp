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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

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
    {
        return mbIsPythonInitializedOK & mbIsYOLACTInitializedOK;
    }
    // 查看错误提示
    inline const std::string& getErrorDescriptionString(void) const
    {
        return mstrErrDescription;
    }

    // 获取网络可以识别的类别数目
    inline size_t getCLassNum(void) const
    {
        return mnClassNum;
    }

    // 获取网络可以识别的类别名称
    // 为了避免破坏封装性，这里要求必须是常值引用
    inline const std::vector<std::string>& getClassNames(void) const 
    {
        return mvstrClassNames;
    }

    // 评估图像
    // bool EvalImage(const cv::Mat& srcImage,
    //                cv::Mat& resImage,
    //                std::vector<size_t>& classId,
    //                std::vector<>
    // )

    
    






private:

    // Python 文件和模块相关
    std::string     mstrPyMoudlePath;
    std::string     mstrPyMoudleName;
    // 保存类型
    size_t mnClassNum;
    std::vector<std::string> mvstrClassNames;
  
    



    // 用于指示错误的
    bool            mbIsYOLACTInitializedOK;
    bool            mbIsPythonInitializedOK;
    std::string     mstrErrDescription;


};      // class YOLACT

}       // namespace YOLACT

#endif  // __YOLACT_HPP__
