/**
 * @file main.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief C++程序和Python程序之间传递图片数据的尝试 -- 增强版： 调用 Yolact 目录下的函数. 
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

// ===================== 正文 =========================
int main(int argc, char* argv[])
{
    cout<<"Test for call python functions in yolact folders."<<endl;
    cout<<"No parameters required."<<endl;
    cout<<"Complied at "<<__TIME__<<" , "<<__DATE__<<" by GNU version "<<__GNUC__<<endl<<endl;


    cout<<"Initlizing Python Environment ..."<<endl;
    init_python_env();
    cout<<"Initlizing Python Environment ... ==> OK."<<endl;

    cv::Mat img=cv::imread("../img/test_pic.jpg");
    if(img.empty())
    {
        cout<<"image is empty!"<<endl;
        return 1;
    }
    else
    {
        cv::imshow("test_pic from c++",img);
        cv::waitKey(0);
    }
    
    
    string py_module_name("eval_cpp_interface");
    // string py_module_name("cpp_simple_test");
    string py_function_name("say_hello");


    if(!run_python_func(py_module_name,py_function_name))
    {
        cout<<"call python function \""<<py_function_name<<"\" in python module \""<<py_module_name<<"\" failed when initializing python environment."<<endl;
        return false;
    }

    // if(!transform_image_to_python(img,"main","disp_img"))
    // {
    //     cout<<"call python function \"init_sys_path()\" failed when initializing python environment."<<endl;
    //     return false;
    // }


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


        


        if(!run_python_func("fix_cv2_sys_path","init_sys_path"))
        {
            cout<<"Calling python function \"init_sys_path()\" failed when initializing python environment."<<endl;
            return false;
        }

        return true;
    }
}

/** @brief 释放python环境 */
void free_python_env(void)
{
    Py_Finalize();
    cout<<"Python Env terminated."<<endl;
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

