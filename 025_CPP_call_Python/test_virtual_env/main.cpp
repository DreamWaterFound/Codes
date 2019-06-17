/**
 * @file main.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 用于验证C++调用Python的时候，虚拟环境是否还可以正常工作
 * @version 0.1
 * @date 2019-06-16
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include <Python.h>

#include <iostream>
#include <string>

using namespace std;

// ======================== 在这里定义函数原型 =============================
// 初始化python环境
bool init_python_env(void);
// 释放Python运行环境
void free_python_env(void);
// 打开moudle，也就是python文件
void load_python_moudle(string py_moudle_name,string py_function_name);


// ============================= 正文 ====================================

/** @brief 主函数 */
int main(int argc, char* argv[])
{
    cout<<"Test if the virtual environment can work when c++ calling python scripts."<<endl;
    cout<<"No parameters required."<<endl;
    cout<<"Complied at "<<__TIME__<<" , "<<__DATE__<<" by GNU version "<<__GNUC__<<endl<<endl;

    cout<<"Tring to initlize python environment ..."<<endl;
    if(init_python_env())
    {
        cout<<"...OjbK."<<endl;
    }
    else
    {
        cout<<"..Failed."<<endl;
    }

    // 调用python文件
    // NOTICE 注意调用的时候是写模块名，也就是不需要带.py后缀
    load_python_moudle("main","say_hello");


    cout<<"Now we are tring to release python environment ..."<<endl;
    free_python_env();
    cout<<"Ok."<<endl;

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
        // if(!(PyRun_SimpleString("import sys"))) 
        // {
        //     cout<<"import sys failed."<<endl;
        //     return false;
        // }
        // if(!(PyRun_SimpleString("sys.path.append('./')")))
        // {
        //     cout<<"sys.path.append('./') failed."<<endl;
        //     return false;
        // }
        // if(!(PyRun_SimpleString("sys.path.append('/home/guoqing/.pyenv/versions/anaconda3-5.2.0/lib/python3.6/site-packages/')")))
        // {
        //     cout<<"append path failed."<<endl;
        //     return false;
        // }

        cout<<PyRun_SimpleString("import sys")<<endl;
        cout<<PyRun_SimpleString("sys.path.append('./')")<<endl;
        // cout<<PyRun_SimpleString("sys.path.append('/home/guoqing/.pyenv/versions/anaconda3-5.2.0/lib/python3.6/site-packages')")<<endl;
        // cout<<PyRun_SimpleString("sys.path.append('/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/tmp_pt/lib/python3.7/site-packages/torch')")<<endl;
        cout<<PyRun_SimpleString("sys.path.append('/home/guoqing/.pyenv/versions/anaconda3-5.2.0/envs/tmp_pt/lib/python3.7/site-packages')")<<endl;
        cout<<PyRun_SimpleString("import matplotlib")<<endl;
        // cout<<PyRun_SimpleString("import cv2")<<endl;

        // 准备解决ROS加的路径问题

        load_python_moudle("main","init_sys_path");

        cout<<"Tring to import cv2 ..."<<endl;
        cout<<PyRun_SimpleString("import cv2")<<endl;

        load_python_moudle("main","show_img");

		cout << "Python Env initialize OK."<<endl;

        
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
 * @param[in] py_moudle_name  文件名
 */
void load_python_moudle(string py_moudle_name,string py_function_name)
{
    PyObject * pModule = nullptr;          //Python模块指针，也就是Python文件
	PyObject * pFunc = nullptr;            //Python中的函数指针
	
	//导入python文件模块
	pModule = PyImport_ImportModule(py_moudle_name.c_str());
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
    // pModule = PyImport_ImportModule("main");
	if(pModule == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pModule "<<py_moudle_name <<" is null"<<endl;
		return ;
	}
    else
    {
        cout<<"Loading python moudle "<<py_moudle_name<<" OK."<<endl;
    }

    // 直接从模块中获取函数指针
    pFunc = PyObject_GetAttrString(pModule, py_function_name.c_str());
    if(pFunc == nullptr)
    {
		cout << "==> Line "<<__LINE__<<":test_python pFunc is null"<<endl;
		return ;
	}

    if(PyEval_CallObject(pFunc, NULL))
    {
        cout<<"==> Line "<<__LINE__<<": python function Ok!"<<endl;
    }
    else
    {
        cout<<"Line "<<__LINE__<<": python function "<<py_function_name<<" in python module "<<py_moudle_name<<" ERROR!"<<endl;
    }
    

    Py_DECREF(pModule);
    
}
