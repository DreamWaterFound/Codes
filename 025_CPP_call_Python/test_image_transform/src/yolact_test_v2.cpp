/**
 * @file main.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief C++程序和Python程序之间传递图片数据的尝试 -- 增强版： 尝试直接调用 YOLACT 进行图像的实例分割
 * @version 0.1
 * @date 2019-06-17
 * 
 * @copyright Copyright (c) 2019
 * 
 */

// =================== 头文件 =========================
#include <Python.h>

#include <iostream>
#include <string>

#include <numpy/arrayobject.h>

#include <opencv2/opencv.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

using namespace std;

// ================ 在这里定义函数原型 ==================
// 初始化python环境
bool init_python_env(void);
// 释放Python运行环境
void free_python_env(void);
// 打开moudle，也就是python文件
bool run_python_func(string py_moudle_name,string py_function_name);
// 将一张图片打包发送到python程序端
bool transform_image_to_python(cv::Mat img,string py_moudle_name, string py_function_name);
// 初始化YOLACT网络
bool init_yolact(string trained_model_path,
                int     top_k               =5,
                bool    cross_class_nms     =true,
                bool    fast_nms            =true,
                bool    display_masks       =true,
                bool    display_bboxes      =true,
                bool    display_text        =true,
                bool    display_scores      =true,
                bool    display_lincomb     =false,
                bool    mask_proto_debug    =false,
                float   score_threshold     =0,
                bool    detect              =false);

// ===================== 正文 =========================
int main(int argc, char* argv[])
{
    cout<<"Test for call yolact to predict image."<<endl;
    cout<<"No parameters required."<<endl;
    cout<<"Complied at "<<__TIME__<<" , "<<__DATE__<<" by GNU version "<<__GNUC__<<endl<<endl;


    cout<<"Initlizing Python Environment ..."<<endl;
    init_python_env();
    cout<<"Initlizing Python Environment ... ==> OK."<<endl;

    bool res;
    res=init_yolact("/home/guoqing/nets/YOLACT/pre_models/yolact_darknet53_54_800000.pth",
    100,true,true,true,true,true,true,false,false,0.3,false);

    if(res)
    {
        cout<<"C++ YOLACT OK."<<endl;
    }
    else
    {
        cout<<"C++ YOLACT Failed."<<endl;
    }

    // cv::Mat img=cv::imread("../img/test_pic.jpg");
    cv::Mat img=cv::imread("/home/guoqing/Datasets/TUM_RGBD/fr3/sitting_static/rgb/1341845688.629688.png");

    
    if(img.empty())
    {
        cout<<"image is empty!"<<endl;
        return 1;
    }
    else
    {
        cv::imshow("test_pic from c++",img);
        cv::waitKey(500);
    }

    cout<<"Predicting Picture ,wait ...."<<endl;
    
    string py_module_name("eval_cpp_interface");
    string py_function_name("evalimage");
    if(!transform_image_to_python(img,py_module_name,py_function_name))
    {
        cout<<"call python function \""<<py_function_name<<"\" in python module \""<<py_module_name<<"\" failed when initializing python environment."<<endl;
        return false;
    }


    cout<<"Deinitlizing Python Environment ..."<<endl;
    free_python_env();
    cout<<"Deinitlizing Python Environment ... ==> OK."<<endl;


    return 0;
}


/**
 * @brief 初始化python环境
 * @return true 
 * @return false 
 */
bool init_python_env(void)
{
    Py_Initialize();
    if(!Py_IsInitialized()){
		cout << "Python Env initialize failed"<<endl;
        return false;
	}
    else
    {
        if(PyRun_SimpleString("import sys")!=0)
        {
            cout<<"\"import sys\" failed when initialize python environment."<<endl;
            return false;
        }

        
        if(PyRun_SimpleString("sys.path.append('./')")!=0)
        {
            cout<<"\"sys.path.append('./')\" failed when initializing python environment."<<endl;
            return false;
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

        return true;
    }
}

/** @brief 释放python环境 */
void free_python_env(void)
{
    Py_Finalize();
    // cout<<"Python Env terminated."<<endl;
}

/**
 * @brief 调用（打开）某个python文件
 * @param[in] py_moudle_name        模块名称，相当于文件名
 * @param[in] py_function_name      python文件中的函数名称
 */
bool run_python_func(string py_moudle_name,string py_function_name)
{
    PyObject * pModule = nullptr;          //Python模块指针，也就是Python文件
	PyObject * pFunc = nullptr;            //Python中的函数指针
	
	// step 1 导入python文件模块
	pModule = PyImport_ImportModule(py_moudle_name.c_str());
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
	if(pModule == nullptr)
    {
        cout<<"Error: import python module \""<<py_moudle_name<<"\" failed."<<endl;
		return false;
	}

    // step 2 从python文件中获取对应的函数的指针
    pFunc = PyObject_GetAttrString(pModule, py_function_name.c_str());
    if(pFunc == nullptr)
    {
        cout<<"Error: python function named \""<< py_function_name<<"\" in python module \""<<py_moudle_name<<"\" NOT found."<<endl;
		return false;
	}

    // step 3 如果函数存在，那么就调用它
    if(!PyEval_CallObject(pFunc, NULL))
    {
        // cout<<"==> Line "<<__LINE__<<": python function Ok!"<<endl;
        cout<<"Error: run the python function named \""<< py_function_name<<"\" in python module \""<<py_moudle_name<<"\" failed."<<endl;
        Py_DECREF(pModule);
        return false;
    }
    else
    {
        // step 4 释放
        Py_DECREF(pModule);
        return true;
    }    
}

/**
 * @brief 将一张图片打包发送到python程序端
 * @param[in] img                   需要发送的图像
 * @param[in] py_moudle_name        接受函数所在的python文件模块
 * @param[in] py_function_name      接受图像的函数
 * @return true                     传送成功
 * @return false                    传送失败
 */
bool transform_image_to_python(cv::Mat img,string py_moudle_name, string py_function_name)
{
    bool res=false;
    // 导入numpy数组格式支持
    import_array();

    // step 0 构造图像数组
    if(img.empty()) return false;

    // 获取图像尺寸
    int x=img.size().width,
        y=img.size().height,
        z=img.channels();
    
    //? 用于保存啥的数组啊
    // NOTICE 记得使用完成之后释放它
    unsigned char *CArrays=new unsigned char[x*y*z];

    // 针对图像的长宽高又获取了一遍
    int iChannels = img.channels();
    int iRows = img.rows;
    int iCols = img.cols * iChannels;
    // 判断这个图像是否是连续存储的，如果是连续存储的，那么意味着我们可以把它看成是一个一维数组，从而加速存取速度
    if (img.isContinuous())
    {
        iCols *= iRows;
        iRows = 1;
    }

    // NOTICE 目前这段程序来看,只能够处理连续空间

    // 指向图像中某个像素所在行的指针
    unsigned char* p;
    // 在每一行中的元素索引
    int id = -1;
    for (int i = 0; i < iRows; i++)
    {
        // get the pointer to the ith row -- 指向当前所遍历到的行
        p = img.ptr<uchar>(i);
        // operates on each pixel
        for (int j = 0; j < iCols; j++)
        {
            CArrays[++id] = p[j];//连续空间
        }
    }

    // 构造numpy数组
    npy_intp Dims[3] = { y, x, z}; //注意这个维度数据！
    PyObject *PyArray = PyArray_SimpleNewFromData(3,            // 有几个维度
                                                  Dims,         // 数组在每个维度上的尺度
                                                  NPY_UBYTE,    // numpy数组中每个元素的类型
                                                  CArrays);     // 用于构造numpy数组的初始数据

    // 由于我们在python文件中使用的函数只有一个参数,所以这里构造的元组也就只有一个元素
    PyObject *ArgList = PyTuple_New(1);
    PyTuple_SetItem(ArgList, 0, PyArray);

    // step 2 加载python模块
    PyObject * pModule = nullptr;          //Python模块指针，也就是Python文件
	PyObject * pFunc = nullptr;            //Python中的函数指针
	
	pModule = PyImport_ImportModule(py_moudle_name.c_str());
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
	if(pModule == nullptr)
    {
        cout<<"Error: import python module \""<<py_moudle_name<<"\" failed."<<endl;
        if(CArrays)
            delete CArrays;
		return false;
	}

    // step 2 从python文件中获取对应的函数的指针
    pFunc = PyObject_GetAttrString(pModule, py_function_name.c_str());
    if(pFunc == nullptr)
    {
        cout<<"Error: python function named \""<< py_function_name<<"\" in python module \""<<py_moudle_name<<"\" NOT found."<<endl;
        if(CArrays)
            delete CArrays;
		return false;
	}

    // step 3 如果函数存在，那么就调用它
    if(!PyEval_CallObject(pFunc, ArgList))
    {
        // cout<<"==> Line "<<__LINE__<<": python function Ok!"<<endl;
        cout<<"Error: run the python function named \""<< py_function_name<<"\" in python module \""<<py_moudle_name<<"\" failed."<<endl;
        Py_DECREF(pModule);
        if(CArrays)
            delete CArrays;
        return false;
    }
    else
    {
        // step 4 释放
        Py_DECREF(pModule);
        if(CArrays)
            delete CArrays;
        return true;
    }
}

// 初始化YOLACT网络；这个函数目前必须要在初始化了python环境之后调用
bool init_yolact(string trained_model_path,
                int     top_k               ,
                bool    cross_class_nms     ,
                bool    fast_nms            ,
                bool    display_masks       ,
                bool    display_bboxes      ,
                bool    display_text        ,
                bool    display_scores      ,
                bool    display_lincomb     ,
                bool    mask_proto_debug    ,
                float   score_threshold     ,
                bool    detect              )
{
    // step 0 构造函数参数
    PyObject *pArgs = PyTuple_New(12);
    PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", trained_model_path.c_str()));
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", top_k));
    PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", cross_class_nms));
    PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", fast_nms));
    PyTuple_SetItem(pArgs, 4, Py_BuildValue("i", display_masks));
    PyTuple_SetItem(pArgs, 5, Py_BuildValue("i", display_bboxes));
    PyTuple_SetItem(pArgs, 6, Py_BuildValue("i", display_text));
    PyTuple_SetItem(pArgs, 7, Py_BuildValue("i", display_scores));
    PyTuple_SetItem(pArgs, 8, Py_BuildValue("i", display_lincomb));
    PyTuple_SetItem(pArgs, 9, Py_BuildValue("i", mask_proto_debug));
    PyTuple_SetItem(pArgs, 10, Py_BuildValue("f", score_threshold));
    PyTuple_SetItem(pArgs, 11, Py_BuildValue("i", detect));

    PyObject * pModule = nullptr;          //Python模块指针，也就是Python文件
	PyObject * pFunc = nullptr;            //Python中的函数指针

    string py_moudle_name("eval_cpp_interface");
    string py_function_name("init_yolact");
	
	// step 1 导入python文件模块
	pModule = PyImport_ImportModule(py_moudle_name.c_str());
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
	if(pModule == nullptr)
    {
        cout<<"Error: import python module \""<<py_moudle_name<<"\" failed."<<endl;
		return false;
	}

    // step 2 从python文件中获取对应的函数的指针
    pFunc = PyObject_GetAttrString(pModule, py_function_name.c_str());
    if(pFunc == nullptr)
    {
        cout<<"Error: python function named \""<< py_function_name<<"\" in python module \""<<py_moudle_name<<"\" NOT found."<<endl;
		return false;
	}

    // step 3 如果函数存在，那么就调用它
    if(!PyEval_CallObject(pFunc, pArgs))
    {
        // cout<<"==> Line "<<__LINE__<<": python function Ok!"<<endl;
        cout<<"Error: run the python function named \""<< py_function_name<<"\" in python module \""<<py_moudle_name<<"\" failed."<<endl;
        Py_DECREF(pModule);
        return false;
    }
    else
    {
        // step 4 释放
        Py_DECREF(pModule);
        return true;
    }    
}

