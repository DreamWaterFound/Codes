/**
 * @file yolact.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief YOLACT操作类的实现
 * @version 0.1
 * @date 2019-06-27
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "yolact.hpp"

namespace YOLACT
{

// 构造函数
YOLACT::YOLACT( 
            std::string pyEnvPkgsPath,
            std::string pyMoudlePathAndName,
            std::string initPyFunctionName,
            std::string evalPyfunctionName,
            std::string trainedModelPath,
            float       scoreThreshold,
            int         topK          ,
            bool        detect        ,
            bool        crossClassNms ,
            bool        fastNms       ,
            bool        displayMasks  ,
            bool        displayBBoxes ,
            bool        displayText   ,
            bool        displayScores ,
            bool        displayLincomb,
            bool        maskProtoDebug)
    :mstrEvalPyfunctionName(evalPyfunctionName),
     mbIsYOLACTInitializedOK(false),
     mbIsPythonInitializedOK(false),
     mpb8ImgTmpArray(nullptr),
     mpPyEvalModule(nullptr),
     mpPyEvalFunc(nullptr)
{
    // step 0 解析路径
    size_t slashPosition=pyMoudlePathAndName.find_last_of("/\\");
    if(slashPosition + 1 == pyMoudlePathAndName.size())
    {
        mstrErrDescription=std::string("Please specified python moudle name.");
        return ;
    }
    size_t pointPosition=pyMoudlePathAndName.find_last_of(".");
    // 这边就先不检查了
    mstrPyMoudlePath=pyMoudlePathAndName.substr(0,slashPosition+1);
    mstrPyMoudleName=pyMoudlePathAndName.substr(slashPosition+1,pointPosition-slashPosition-1);

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

    // step 2 设置 Python 环境，为导入 YOLACT 的程序做准备
    if(PyRun_SimpleString("import sys")!=0)
    {
        mstrErrDescription=std::string("\"import sys\" failed when initialize python environment.");
        return ;
    }

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

    sstrCommand.str("");
    sstrCommand.clear();
    sstrCommand<<"sys.path.append('";
    sstrCommand<<pyEnvPkgsPath;
    sstrCommand<<"')";

    if(PyRun_SimpleString((sstrCommand.str()).c_str())!=0)
    {
        sstrCommand<<" failed.";
        mstrErrDescription=sstrCommand.str();
        return ;
    }

    // step 3 Python环境初始化完成，准备初始化YOLACT网络。构造函数参数
    PyObject *pArgs = PyTuple_New(12);
    PyTuple_SetItem(pArgs, 0,  Py_BuildValue("s", trainedModelPath.c_str()));
    PyTuple_SetItem(pArgs, 1,  Py_BuildValue("i", topK));
    PyTuple_SetItem(pArgs, 2,  Py_BuildValue("i", crossClassNms));
    PyTuple_SetItem(pArgs, 3,  Py_BuildValue("i", fastNms));
    PyTuple_SetItem(pArgs, 4,  Py_BuildValue("i", displayMasks));
    PyTuple_SetItem(pArgs, 5,  Py_BuildValue("i", displayBBoxes));
    PyTuple_SetItem(pArgs, 6,  Py_BuildValue("i", displayText));
    PyTuple_SetItem(pArgs, 7,  Py_BuildValue("i", displayScores));
    PyTuple_SetItem(pArgs, 8,  Py_BuildValue("i", displayLincomb));
    PyTuple_SetItem(pArgs, 9,  Py_BuildValue("i", maskProtoDebug));
    PyTuple_SetItem(pArgs, 10, Py_BuildValue("f", scoreThreshold));
    PyTuple_SetItem(pArgs, 11, Py_BuildValue("i", detect));

    
	PyObject * pFunc = nullptr;            //Python中的函数指针

    // step 4 导入python文件模块
    // 导入numpy数组格式支持
    if(!ImportNumPySupport())
    {
        sstrCommand.str("");
        sstrCommand.clear();
        sstrCommand<<"Error: import numpy support failed.";
        mstrErrDescription=sstrCommand.str();

        return ;
    }
    
    mpPyEvalModule = PyImport_ImportModule(mstrPyMoudleName.c_str());
    
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
	if(mpPyEvalModule == nullptr)
    {
        sstrCommand.str("");
        sstrCommand.clear();
        sstrCommand<<"Error: import python module \"";
        sstrCommand<<mstrPyMoudleName;
        sstrCommand<<"\" failed.";
        mstrErrDescription=sstrCommand.str();

		return ;
	}


    // step 5 获取Python文件中对应的函数指针
    pFunc = PyObject_GetAttrString(mpPyEvalModule, initPyFunctionName.c_str());
    if(pFunc == nullptr)
    {
        sstrCommand.str("");
        sstrCommand.clear();
        sstrCommand<<"Error: YOLACT initlizing function named \"";
        sstrCommand<<initPyFunctionName;
        sstrCommand<<"\" in python module \"";
        sstrCommand<<mstrPyMoudleName;
        sstrCommand<<"\" NOT found.";
        mstrErrDescription=sstrCommand.str();
		return ;
	}

    // step 6 如果函数存在，那么就调用它
    PyObject* pRet=nullptr;
    pRet=PyEval_CallObject(pFunc, pArgs);
    if(!pRet)
    {
        sstrCommand.str("");
        sstrCommand.clear();
        sstrCommand<<"Error: run the python function named \"";
        sstrCommand<<initPyFunctionName;
        sstrCommand<<"\" in python module \"";
        sstrCommand<<mstrPyMoudleName;
        sstrCommand<<"\" failed.";
        mstrErrDescription=sstrCommand.str();
		return ;
    }

    // step 7 获取类别信息
    // 检查返回值是否正常
    if(!PyTuple_Check(pRet))
    {
        mstrErrDescription=std::string("YOLACT init function get wrong ret-value.");
        return ;
    }

    // 获取类型数目
    mnClassNum=PyTuple_GET_SIZE(pRet);
    if(mnClassNum==0)
    {
        mstrErrDescription=std::string("0 classes get. This shouldn't be happened.");
        return ;
    }

    // 确定类型
    PyObject *pStr=PyTuple_GetItem(pRet,0);
    if(!PyUnicode_Check(pStr))
    {
        mstrErrDescription=std::string("Ret-value tuple DOESN'T have an unicode strings.");
        return ;
    }
    if(PyUnicode_GET_LENGTH(pStr)==0)
    {
        mstrErrDescription=std::string("Ret-value tuple[0] is non string.");
        return ;
    }
    // 原则上这里需要根据不同的类型调用不同的解析函数，但是现在嫌费劲就不做了
    if(PyUnicode_KIND(pStr)!=PyUnicode_1BYTE_KIND)
    {
        mstrErrDescription=std::string("Ret-value tuple[0]'s type is not PyUnicode_1BYTE_KIND.");
        return ;
    }

    // 都确认了之后，就可以批量处理了
    mvstrClassNames.clear();
    for(size_t i=0;i<mnClassNum;++i)
    {
        mvstrClassNames.emplace_back((char*)PyUnicode_1BYTE_DATA(PyTuple_GetItem(pRet,i)));
    } 

    // // step 8 释放 // ? 感觉这里还需要补全
    // // ! 而且之前出现的内存占用非常多的问题，估计是和没有能够释放一些东西有关系
    // Py_DECREF(pModule);   

    mbIsYOLACTInitializedOK=true;


    // DEBUG 
    std::cout<<"Final OK."<<std::endl;


}

// 析构函数
YOLACT::~YOLACT()
{
    if(mbIsPythonInitializedOK)
    {
        std::cout<<"Exiting python env ..."<<std::endl;
        Py_Finalize();
    }

    if(mpb8ImgTmpArray)
    {
        delete mpb8ImgTmpArray;
    }

    if(mpPyEvalModule)
    {
        Py_DecRef(mpPyEvalModule);
    }

    if(mpPyEvalFunc)
    {
        Py_DecRef(mpPyEvalFunc);
    }
}

bool YOLACT::EvalImage(const cv::Mat& srcImage,
                cv::Mat& resImage,
                std::vector<std::string>& vstrClassName,
                std::vector<float>& vdScores,
                std::vector<std::pair<cv::Point2i,cv::Point2i> >& vpairBBoxes,
                std::vector<cv::Mat>& vimgMasks)
{
    std::vector<size_t> vnClassId;
    bool res=EvalImage(srcImage,resImage,vnClassId,vdScores,vpairBBoxes,vimgMasks);
    vstrClassName.clear();
    for(int i=0;i<vnClassId.size();++i)
    {
        vstrClassName.push_back(mvstrClassNames[vnClassId[i]]);
    }

    return res;
}

bool YOLACT::EvalImage(const cv::Mat& srcImage,
                cv::Mat& resImage,
                std::vector<size_t>& vstrClassId,
                std::vector<float>& vdScores,
                std::vector<std::pair<cv::Point2i,cv::Point2i> >& vpairBBoxes,
                std::vector<cv::Mat>& vimgMasks)
{
    // step 1 将图片转换成为NumPy的数组的形式
    PyObject *pPyImageArray=nullptr;
    if(!Image2Numpy(srcImage,pPyImageArray))
    {
        // 错误字符串的生成已经在上面的函数中进行了
        return ;
    }

    // step 2 构造函数参数
    PyObject *pPyArgList = PyTuple_New(1);
    PyTuple_SetItem(pPyArgList, 0, pPyImageArray);

    // step 3 获取 Python 端函数指针
    if(!mpPyEvalModule)
    {
        mstrErrDescription=std::string("Python Moudle pointer ERROR.");
        return false;
    }
    // 获取评估函数指针
    if(!mpPyEvalFunc)
    {
        mpPyEvalFunc=PyObject_GetAttrString(mpPyEvalModule, mstrEvalPyfunctionName.c_str());
        if(mpPyEvalFunc==nullptr)
        {
            std::stringstream ss;
            ss<<"Error: YOLACT initlizing function named \"";
            ss<<mstrEvalPyfunctionName;
            ss<<"\" in python module \"";
            ss<<mstrPyMoudleName;
            ss<<"\" NOT found.";
            mstrErrDescription=ss.str();
            return false;
        }
    }

    // step 5 现在说明这个函数的确存在，那么我们就调用它
    PyObject* pPyRetValue=nullptr;
    pPyRetValue=PyEval_CallObject(mpPyEvalModule,pPyArgList);

    // step 6 返回值的初步检查和初步解析
    if(!pPyRetValue)
    {
        std::stringstream ss;
        ss<<"Error occured when calling method \"";
        ss<<mstrEvalPyfunctionName;
        ss<<"\" in python module \"";
        ss<<mstrPyMoudleName;
        ss<<"\". ";
        mstrErrDescription=ss.str();
        return false;
    }

    if(!PyTuple_Check(pPyRetValue))
    {
        mstrErrDescription=std::string("Eval image function did NOT return a tuple.");
        return false;
    }

    if(PyTuple_Size(pPyRetValue)!=5)
    {
        mstrErrDescription=std::string("Eval image function did NOT return a tuple with correct items.");
        return false;
    }

    // 存储解析后的数据
    PyArrayObject *pClasses,*pScores,*pBoxes,*pMasks,*pResImgArray;

    pClasses        = (PyArrayObject*)PyTuple_GetItem(pPyRetValue,0);
    pScores         = (PyArrayObject*)PyTuple_GetItem(pPyRetValue,1);
    pBoxes          = (PyArrayObject*)PyTuple_GetItem(pPyRetValue,2);
    pMasks          = (PyArrayObject*)PyTuple_GetItem(pPyRetValue,3);
    pResImgArray    = (PyArrayObject*)PyTuple_GetItem(pPyRetValue,4);

    // step 7 解析类别id数据
    size_t lenOfClassesId=pClasses->dimensions[0];
    vstrClassId.clear();
    for(size_t i=0;i<lenOfClassesId;++i)
    {
        vstrClassId.push_back(
            *(size_t*)(pClasses->data+i*(pClasses->strides[0]))
        );
    }

    // step 8 解析类别评分数据
    size_t lenOfScorces=pScores->dimensions[0];
    vdScores.clear();
    for(size_t i=0;i<lenOfScorces;++i)
    {
        vdScores.push_back(
            *(float*)(pScores->data+i*(pScores->strides[0]))
        );
    }

    // step 9 TODO 

    // step 10 TODO 


    // step ？处理结果图像
    // TODO 其实这里可以在增加处理一个能够不处理这个结果图像的函数 --- 这样就可以加快处理速度了

    size_t resImageH=pResImgArray->dimensions[0],
           resImageW=pResImgArray->dimensions[1];

    resImage=cv::Mat(resImageH,resImageW,CV_8UC3,pResImgArray->data);





    








}

bool YOLACT::Image2Numpy(const cv::Mat& srcImage,
                PyObject *pPyArray) 
{
    // step 0 检查图像是否非空
    if(srcImage.empty())
    {
        mstrErrDescription=std::string("src image is empty!");
        return false;
    }

    // step 1 生成临时的图像数组，图像数组暂时是缓存在成员变量里。检查合法性
    if(mpb8ImgTmpArray)
    {
        delete mpb8ImgTmpArray;
    }

    // 获取图像的尺寸
    size_t x=srcImage.size().width,
           y=srcImage.size().height,
           z=srcImage.channels();
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



}       // namespace YOLACT

