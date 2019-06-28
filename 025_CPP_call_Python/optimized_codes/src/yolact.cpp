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
    :mbIsYOLACTInitializedOK(false),
     mbIsPythonInitializedOK(false)
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

    PyObject * pModule = nullptr;          //Python模块指针，也就是Python文件
	PyObject * pFunc = nullptr;            //Python中的函数指针

    // step 4 导入python文件模块
    pModule = PyImport_ImportModule(mstrPyMoudleName.c_str());
    
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
	if(pModule == nullptr)
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
    pFunc = PyObject_GetAttrString(pModule, initPyFunctionName.c_str());
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

    // step 8 释放 // ? 感觉这里还需要补全
    // ! 而且之前出现的内存占用非常多的问题，估计是和没有能够释放一些东西有关系
    Py_DECREF(pModule);   

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
}



}       // namespace YOLACT

