/**
 * @file main.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 尝试调用一个Python函数并且获取其返回值
 * @version 0.1
 * @date 2019-06-21
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
// bool run_python_func(string py_moudle_name,string py_function_name);
// 这个是无参数但是却带有返回值的
bool run_python_func_with_value(string py_moudle_name,string py_function_name);

// ===================== 正文 =========================
int main(int argc, char* argv[])
{
    cout<<"Test for call yolact to predict image."<<endl;
    cout<<"No parameters required."<<endl;
    cout<<"Complied at "<<__TIME__<<" , "<<__DATE__<<" by GNU version "<<__GNUC__<<endl<<endl;


    cout<<"Initlizing Python Environment ..."<<endl;
    init_python_env();
    cout<<"Initlizing Python Environment ... ==> OK."<<endl;

    cout<<"Testing the python return value, wait ...."<<endl;
    
    // TODO 修改它
    string py_module_name("py_return_test");
    string py_function_name("get_value");
    if(!run_python_func_with_value(py_module_name,py_function_name))
    {
        cout<<"call python function \""<<py_function_name<<"\" in python module \""<<py_module_name<<"\" failed."<<endl;
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
 * @brief 调用（打开）某个python文件,这个调用的函数具有返回值
 * @param[in] py_moudle_name        模块名称，相当于文件名
 * @param[in] py_function_name      python文件中的函数名称
 */
bool run_python_func_with_value(string py_moudle_name,string py_function_name)
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

    // step 3 如果函数存在，那么就调用它 （这里是无参数）
    // 如果Python函数存在返回值,那么只能够通过判断解析返回值是否成功来判断了
    PyObject* pRet = PyEval_CallObject(pFunc,nullptr);
    unsigned char res[1024]={0x00};
    // string str;
    int a;

    // if(pRet && PyArg_ParseTuple(pRet,"si",str,&a))
    if(pRet && PyArg_Parse(pRet,"si", res,&a))
    // if(pRet)
    {
        printf("%s\n",res);
        // cout<<"str="<<str<<endl;
        cout<<"a="<<a<<endl;

        cout<<"Ok!"<<endl;


        // step 4 释放
        Py_DECREF(pModule);
        return true;   
    }
    else
    {
        cout<<"Error: run the python function named \""<< py_function_name<<"\" in python module \""<<py_moudle_name<<"\" failed."<<endl;
        Py_DECREF(pModule);
        return false;
    }    
}
