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
            bool        maskProtoDebug     =false)
    :mbIsYOLACTInitializedOK(false),
     mbIsPythonInitializedOK(false)
{
    // step 0 解析路径
    size_t positionFound=pyMoudlePathAndName.find_last_of("/\\");
    if(positionFound + 1 == pyMoudlePathAndName.size())
    {
        mstrErrDescription=std::string("Please specified python moudle name.");
        return ;
    }
    mstrPyMoudlePath=pyMoudlePathAndName.substr(0,positionFound);
    mstrPyMoudleName=pyMoudlePathAndName.substr(positionFound+1);

    // DEBUG
    std::cout<<"[debug](YOLACT::YOLACT) mstrPyMoudlePath="<<mstrPyMoudlePath<<std::endl;
    std::cout<<"[debug](YOLACT::YOLACT) mstrPyMoudleName="<<mstrPyMoudleName
    <<std::endl;

    // TODO  写到这里的时候其实可以测试一下


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

    if(PyRun_SimpleString("sys.path.append('/home/guoqing/nets/YOLACT/yolact_master-comment/')")!=0)
    {
        cout<<"\"sys.path.append('/home/guoqing/nets/YOLACT/yolact_master-comment/')\" failed when initializing python environment."<<endl;
        return false;
    }

    if(PyRun_SimpleString("sys.path.append('/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/tmp_pt/lib/python3.7/site-packages')")!=0)
    {
        cout<<"\"sys.path.append('/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/tmp_pt/lib/python3.7/site-packages')\" failed when initialize python environment."<<endl;
        return false;
    }


    


}







}       // namespace YOLACT

