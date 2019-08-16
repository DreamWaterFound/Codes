/**
 * @file LEDNet_interface.hpp
 * @author guoqing (1337841346@qq.com)
 * @brief LEDNet的接口函数
 * @version 0.1
 * @date 2019-08-16
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#ifndef __LEDNET_INTERFACE_HPP__
#define __LEDNET_INTERFACE_HPP__

// C++基础
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

// Python 支持
#include <Python.h>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif 

#include "numpy/arrayobject.h"


// OpenCV支持
#include <opencv2/opencv.hpp>

#define EVAL_PY_FUNCTION_NAME       "LEDNet_eval"
#define INIT_PY_FUNCTION_NAME       "LEDNet_init"
#define PY_ENV_PKGS_PATH            "/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/pt041/lib/python3.6/site-packages"


namespace LEDNET
{

class LEDNET
{

public:
    // 构造函数
    LEDNET(const std::string&   pyFilePath,
           const std::string&   modelPath,
           const size_t&        categories);
    // 析构函数
    ~LEDNET();

public:

    // 执行一张图像的预测
    bool evalImage(const cv::Mat&   inputImage,
                         cv::Mat&   confidenceImage,
                         cv::Mat&   labelImage);

public:

     // 查看是否初始化成功
    inline bool isInitializedResult(void) const
    {   return mbIsPythonInitializedOK & mbIsLEDNETInitializedOK;   }

    // 查看错误提示
    inline const std::string& getErrorDescriptionString(void) const
    {   return mstrErrDescription;    }

    // 获取网络可以识别的类别数目
    inline size_t getCLassNum(void) const
    {   return mnCategories;    }

private:

    bool ImportNumPySupport(void) const
    {
        // 这是一个宏，其中包含了返回的语句
        import_array();
        return true;
    }

    bool Image2Numpy(const cv::Mat& srcImage,
                PyObject*& pPyArray);

    bool parseFilePathAndName(const std::string& strFilePathAndName);

private:

    // Python 文件和模块相关
    std::string     mstrPyEnvPkgsPath;              // 指定了 anaconda 寻找文件的路径 site-packages
    std::string     mstrPyMoudlePath;
    std::string     mstrPyMoudleName;
    std::string     mstrEvalPyFunctionName;
    std::string     mstrInitPyFunctionName;
    PyObject*       mpPyEvalModule;
    PyObject*       mpPyEvalFunc;

    // 类别个数
    size_t          mnCategories;

    // 用于在图像转换时的数组指针
    unsigned char  *mpb8ImgTmpArray;

    // 用于指示错误的
    bool            mbIsLEDNETInitializedOK;
    bool            mbIsPythonInitializedOK;
    std::string     mstrErrDescription;

    // 一些中间的文件
    PyObject *mpPyArgList;
    PyObject *mpPyRetValue;
};      // class LEDNET

LEDNET::LEDNET(const std::string&   pyFilePath,
               const std::string&   modelPath,
               const size_t&        categories)
   :mstrPyEnvPkgsPath(PY_ENV_PKGS_PATH),
    mstrEvalPyFunctionName(EVAL_PY_FUNCTION_NAME),
    mstrInitPyFunctionName(INIT_PY_FUNCTION_NAME),
    mpPyEvalModule(nullptr),
    mpPyEvalFunc(nullptr),
    mnCategories(categories),
    mpb8ImgTmpArray(nullptr),
    mbIsLEDNETInitializedOK(false),
    mbIsPythonInitializedOK(false),
    mpPyArgList(nullptr),
    mpPyRetValue(nullptr)
{
    // step 0 解析路径, 解析后的结果将会被存放在类的成员变量中
    if(!parseFilePathAndName(pyFilePath))
    {
        return ;
    }

    // step 1 初始化 Python 环境
    Py_Initialize();
    if(!Py_IsInitialized())
    {
        mstrErrDescription=std::string("Python Env initialize failed.");
        return ;
	}
    else
    {
        mbIsPythonInitializedOK=true;
    }

    // step 2 设置 Python 环境，为导入 LEDNET 的程序做准备
    // 1. 导入 sys 模块
    if(PyRun_SimpleString("import sys")!=0)
    {
        mstrErrDescription=std::string("\"import sys\" failed when initialize python environment.");
        return ;
    }

    // 2. 将 Python 文件所在的目录添加到包的搜索目录中
    std::stringstream sstrCommand;
    sstrCommand<<"sys.path.append('";
    sstrCommand<<mstrPyMoudlePath;
    sstrCommand<<"')";
    if(PyRun_SimpleString((sstrCommand.str()).c_str())!=0)
    {
        sstrCommand<<" failed.";
        mstrErrDescription=sstrCommand.str();
        return ;
    }

    // 3. 将 Anaconda 虚拟环境中的 Python 的 site-packages 目录添加到搜索目录中
    sstrCommand.str("");
    sstrCommand.clear();
    sstrCommand<<"sys.path.append('";
    sstrCommand<<mstrPyEnvPkgsPath;
    sstrCommand<<"')";
    if(PyRun_SimpleString((sstrCommand.str()).c_str())!=0)
    {
        sstrCommand<<" failed.";
        mstrErrDescription=sstrCommand.str();
        return ;
    }

    // 导入numpy数组格式支持
    if(!ImportNumPySupport())
    {
        sstrCommand.str("");
        sstrCommand.clear();
        sstrCommand<<"Error: import numpy support failed.";
        mstrErrDescription=sstrCommand.str();

        return ;
    }

    mbIsPythonInitializedOK=true;

    // step 3 尝试先调用一个简单的函数
    // PyObject * pModule = nullptr;
	mpPyEvalModule = PyImport_ImportModule(mstrPyMoudleName.c_str());
    if(!mpPyEvalModule)
    {
        sstrCommand.str("");
        sstrCommand.clear();
        sstrCommand<<"Error: import py moudle "<<mstrPyMoudleName<<" failed.";
        mstrErrDescription=sstrCommand.str();
        return ;
    }
    PyObject *pFunc = PyObject_GetAttrString(mpPyEvalModule, mstrInitPyFunctionName.c_str());

    if(!pFunc)
    {
        sstrCommand.str("");
        sstrCommand.clear();
        sstrCommand<<"Error: funtc named \""<<mstrInitPyFunctionName<<"\" in py moudle \""<<mstrPyMoudleName<<"\" not found.";
        mstrErrDescription=sstrCommand.str();

        return ;
    }

    PyObject *pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0,  Py_BuildValue("s", modelPath.c_str()));
    PyTuple_SetItem(pArgs, 1,  Py_BuildValue("i", categories));

    if(PyEval_CallObject(pFunc, pArgs)!=nullptr)
    {
        std::cout<<"[LEDNET] Constructor return is not zero."<<std::endl;
    }

    mbIsLEDNETInitializedOK=true;

}

// 从给定的字符串中解析得到访问文件的目录和名字
bool LEDNET::parseFilePathAndName(const std::string& strFilePathAndName)
{
    size_t slashPosition=strFilePathAndName.find_last_of("/\\");
    if(slashPosition + 1 == strFilePathAndName.size())
    {
        mstrErrDescription=std::string("Please specified python moudle name.");
        return false;
    }
    size_t pointPosition=strFilePathAndName.find_last_of(".");
    // 这边就先不检查了
    mstrPyMoudlePath=strFilePathAndName.substr(0,slashPosition+1);
    mstrPyMoudleName=strFilePathAndName.substr(slashPosition+1,pointPosition-slashPosition-1);

    return true;
}

LEDNET::~LEDNET()
{
    if(mbIsPythonInitializedOK)
    {
        std::cout<<"Exiting python env ..."<<std::endl;
        Py_Finalize();
        std::cout<<"Done."<<std::endl;
    }
}

bool LEDNET::evalImage(const cv::Mat&   inputImage,
                         cv::Mat&   confidenceImage,
                         cv::Mat&   labelImage)
{
    // step 1 将图片转换成为NumPy的数组的形式
    PyObject *pPyImageArray=nullptr;
    if(!Image2Numpy(inputImage,pPyImageArray))
    {
        // 错误字符串的生成已经在上面的函数中进行了
        return false;
    }

    if(!pPyImageArray)
    {
        mstrErrDescription=std::string("pPyImageArray shouldn't be empty.");
        return false;
    }

    // step 2 构造函数参数
    if(mpPyArgList==nullptr)
    {
        mpPyArgList=PyTuple_New(1);
    }
    PyTuple_SetItem(mpPyArgList, 0, pPyImageArray);

    // step 3 获取 Python 端函数指针
    if(!mpPyEvalModule)
    {
        mstrErrDescription=std::string("Python Moudle pointer ERROR.");
        return false;
    }

    // 获取评估函数指针
    if(!mpPyEvalFunc)
    {
        mpPyEvalFunc=PyObject_GetAttrString(mpPyEvalModule, mstrEvalPyFunctionName.c_str());
        if(!mpPyEvalFunc)
        {
            std::stringstream ss;
            ss<<"Error: YOLACT function named \"";
            ss<<mstrEvalPyFunctionName;
            ss<<"\" in python module \"";
            ss<<mstrPyMoudleName;
            ss<<"\" NOT found.";
            mstrErrDescription=ss.str();
            return false;
        }
    }

    // step 5 现在说明这个函数的确存在，那么我们就调用它
    if(mpPyRetValue)
    {
        Py_DECREF(mpPyRetValue);
    }
    mpPyRetValue=PyEval_CallObject(mpPyEvalFunc,mpPyArgList);

    // step 6 返回值的初步检查和初步解析
    if(!mpPyRetValue)
    {
        std::stringstream ss;
        ss<<"Error occured when calling method \"";
        ss<<mstrEvalPyFunctionName;
        ss<<"\" in python module \"";
        ss<<mstrPyMoudleName;
        ss<<"\". ";
        mstrErrDescription=ss.str();
        return false;
    }

    if(!PyTuple_Check(mpPyRetValue))
    {
        mstrErrDescription=std::string("Eval image function did NOT return a tuple.");
        return false;
    }

    
    if(PyTuple_Size(mpPyRetValue)!=2)
    {
        mstrErrDescription=std::string("Eval image function did NOT return a tuple with correct items.");
        return false;
    }

    PyArrayObject *pConfidenceImg, *pLabelImg;

    pConfidenceImg  = (PyArrayObject*)PyTuple_GetItem(mpPyRetValue,0);
    pLabelImg       = (PyArrayObject*)PyTuple_GetItem(mpPyRetValue,1);

    size_t cfdImageH=PyArray_DIMS(pConfidenceImg)[0],
           cfdImageW=PyArray_DIMS(pConfidenceImg)[1];
    confidenceImage=cv::Mat(cfdImageH,cfdImageW,CV_32FC1,PyArray_DATA(pConfidenceImg));

    size_t labelImageH=PyArray_DIMS(pConfidenceImg)[0],
           labelImageW=PyArray_DIMS(pConfidenceImg)[1];
    labelImage=cv::Mat(labelImageH,labelImageW,CV_8UC1,PyArray_DATA(pLabelImg));

    return true;
}                       

bool LEDNET::Image2Numpy(const cv::Mat& srcImage,
                PyObject*& pPyArray) 
{
    // step 0 检查图像是否非空
    if(srcImage.empty())
    {
        mstrErrDescription=std::string("src image is empty!");
        return false;
    }

    // step 1 生成临时的图像数组，图像数组暂时是缓存在成员变量里。检查合法性
    // 获取图像的尺寸
    size_t x=srcImage.size().width,
           y=srcImage.size().height,
           z=srcImage.channels();

    if(mpb8ImgTmpArray)
    {
        delete mpb8ImgTmpArray;
    }
    // 生成
    mpb8ImgTmpArray=new unsigned char[x*y*z];

    size_t iChannels = srcImage.channels(),
           iRows     = srcImage.rows,
           iCols     = srcImage.cols * iChannels;

    // 判断这个图像是否是连续存储的，如果是连续存储的，那么意味着我们可以把它看成是一个一维数组，从而加速存取速度
    if (srcImage.isContinuous())
    {
        iCols *= iRows;
        iRows = 1;
    }

    // 指向图像中某个像素所在行的指针
    unsigned char* p;
    // 在每一行中的元素索引
    int id = -1;
    for (int i = 0; i < iRows; i++)
    {
        // get the pointer to the ith row -- 指向当前所遍历到的行
        p = (unsigned char*)srcImage.ptr<unsigned char>(i);
        // operates on each pixel
        for (int j = 0; j < iCols; j++)
        {
            mpb8ImgTmpArray[++id] = p[j];//连续空间
        }
    }

    // step 2 生成三维的numpy
    npy_intp Dims[3] = {(int)y, 
                        (int)x, 
                        (int)z}; //注意这个维度数据！
    pPyArray = PyArray_SimpleNewFromData(
        3,                      // 有几个维度
        Dims,                   // 数组在每个维度上的尺度
        NPY_UBYTE,              // numpy数组中每个元素的类型
        mpb8ImgTmpArray);       // 用于构造numpy数组的初始数据

    return true;
}


}       // namespace LEDNET




#endif  // __LEDNET_INTERFACE_HPP__