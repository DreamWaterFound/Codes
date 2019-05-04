/**
 * @file main.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief C++ 调用Python的尝试
 * @version 0.1
 * @date 2019-05-04
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include <Python.h>
#include <iostream>
#include <string>

using namespace std;

// 初始化Python运行环境
bool init_python_env(void);
// 释放Python运行环境
void free_python_env(void);
// 调用无参数的函数
void call_py_hello(string py_moudle_name);
// 调用参数为字符串的函数
void call_py_greet_user(string py_moudle_name,string user_name);
// 调用具有两个参数，并且参数类型为数字的函数
void call_py_add_2_num(string py_moudle_name,int n1,int n2);
// 调用具有两个参数并且具有返回值的函数
void call_py_times_2_num(string py_moudle_name,int n1,int n2);
// 调用返回为字符串类型值的函数
void call_py_get_user_name(string py_moudle_name);

// 调用类
void call_py_stu_class_test(string py_moudle_name);

int main(int argc,char *argv[])
{
    cout<<"A test for C++ call python."<<endl;
    cout<<"Complied at "<<__TIME__<<" , "<<__DATE__<<endl<<endl;

    if(argc!=2)
    {
        cout<<"Usage: "<<argv[0]<<" python_moudle_name"<<endl;
        cout<<"Ex: "<<argv[0]<<" main"<<endl;
        return 0;
    }

    if(!init_python_env())  return 0;
    
    cout<<"test 01: "<<endl;
    call_py_hello(argv[1]);

    cout<<endl<<"test 02: "<<endl;
    
    call_py_greet_user(argv[1],"Guoqing Liu");

    cout<<endl<<"test 03: "<<endl;

    call_py_add_2_num(argv[1],10,20);

    cout<<endl<<"test 04: "<<endl;

    call_py_times_2_num(argv[1],6,7);

    cout<<endl<<"test 05: "<<endl;
    call_py_get_user_name(argv[1]);

    cout<<endl<<"test 06: "<<endl;
    call_py_stu_class_test(argv[1]);

    free_python_env();

    return 0;
}

bool init_python_env(void)
{
    Py_Initialize();
    if(!Py_IsInitialized()){
		cout << "Python Env initialize failed"<<endl;
        return false;
	}
    else
    {
		cout << "Python Env initialize OK."<<endl;
        PyRun_SimpleString("import sys");
    	PyRun_SimpleString("sys.path.append('./')");
        return true;
    }
}

void free_python_env(void)
{
    Py_Finalize();
    cout<<"Python Env terminated."<<endl;
}

void call_py_hello(string py_moudle_name)
{
    PyObject * pModule = nullptr;          //Python模块指针，也就是Python文件
	PyObject * pFunc = nullptr;            //Python中的函数指针
	
	//导入python文件模块
	pModule = PyImport_ImportModule(py_moudle_name.c_str());
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
    // pModule = PyImport_ImportModule("main");
	if(pModule == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pModule is null"<<endl;
		return ;
	}
    else
    {
        cout<<"Loading python moudle "<<py_moudle_name<<" OK."<<endl;
    }

    // 直接从模块中获取函数指针
    pFunc = PyObject_GetAttrString(pModule, "say_hello");
    if(pFunc == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pFunc is null"<<endl;
		return ;
	}

    PyEval_CallObject(pFunc, NULL);

    Py_DECREF(pModule);
    
}

void call_py_greet_user(string py_moudle_name,string user_name)
{
    PyObject * pModule = nullptr;          //Python模块指针，也就是Python文件
	PyObject * pFunc = nullptr;            //Python中的函数指针
	
	//导入python文件模块
	pModule = PyImport_ImportModule(py_moudle_name.c_str());
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
	if(pModule == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pModule is null"<<endl;
		return ;
	}
    else
    {
        cout<<"Loading python moudle "<<py_moudle_name<<" OK."<<endl;
    }

    // 直接从模块中获取函数指针
    pFunc = PyObject_GetAttrString(pModule, "greet_user");
    if(pFunc == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pFunc is null"<<endl;
		return ;
	}

    PyObject *pArg = Py_BuildValue("(s)", user_name.c_str());
    PyEval_CallObject(pFunc, pArg);

    Py_DECREF(pModule);
}

void call_py_add_2_num(string py_moudle_name,int n1,int n2)
{
    PyObject * pModule = nullptr;          //Python模块指针，也就是Python文件
	PyObject * pFunc = nullptr;            //Python中的函数指针
	
	//导入python文件模块
	pModule = PyImport_ImportModule(py_moudle_name.c_str());
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
	if(pModule == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pModule is null"<<endl;
		return ;
	}
    else
    {
        cout<<"Loading python moudle "<<py_moudle_name<<" OK."<<endl;
    }

    // 直接从模块中获取函数指针
    pFunc = PyObject_GetAttrString(pModule, "add_2_numbers");
    if(pFunc == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pFunc is null"<<endl;
		return ;
	}

    PyObject *pArgs = PyTuple_New(2);//函数调用的参数传递均是以元组的形式打包的,2表示参数个数
	PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", n1));//0--序号,i表示创建int型变量，下标也是从0开始
	PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", n2));//1--序号

    PyEval_CallObject(pFunc, pArgs);

    Py_DECREF(pModule);
}

void call_py_times_2_num(string py_moudle_name,int n1,int n2)
{
    PyObject * pModule = nullptr;          //Python模块指针，也就是Python文件
	PyObject * pFunc = nullptr;            //Python中的函数指针
	
	//导入python文件模块
	pModule = PyImport_ImportModule(py_moudle_name.c_str());
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
	if(pModule == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pModule is null"<<endl;
		return ;
	}
    else
    {
        cout<<"Loading python moudle "<<py_moudle_name<<" OK."<<endl;
    }

    // 直接从模块中获取函数指针
    pFunc = PyObject_GetAttrString(pModule, "multi_2_numbers");
    if(pFunc == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pFunc is null"<<endl;
		return ;
	}

    PyObject *pArgs = PyTuple_New(2);//函数调用的参数传递均是以元组的形式打包的,2表示参数个数
	PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", n1));//0--序号,i表示创建int型变量，下标也是从0开始
	PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", n2));//1--序号

    PyObject *pReturn = nullptr;
    pReturn = PyEval_CallObject(pFunc, pArgs);
    if(pReturn == nullptr)
    {
        cout<<"Return data is null."<<endl;
        return ;
    }

    int result;
    PyArg_Parse(pReturn, "i", &result);//i表示转换成c++ int型变量
    cout<<"C++: "<<n1<<" * "<<n2<<" = "<<result<<endl;
    Py_DECREF(pModule);
}

void call_py_get_user_name(string py_moudle_name)
{
    PyObject * pModule = nullptr;          //Python模块指针，也就是Python文件
	PyObject * pFunc = nullptr;            //Python中的函数指针
	
	//导入python文件模块
	pModule = PyImport_ImportModule(py_moudle_name.c_str());
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
	if(pModule == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pModule is null"<<endl;
		return ;
	}
    else
    {
        cout<<"Loading python moudle "<<py_moudle_name<<" OK."<<endl;
    }

    // 直接从模块中获取函数指针
    pFunc = PyObject_GetAttrString(pModule, "get_user_name");
    if(pFunc == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pFunc is null"<<endl;
		return ;
	}

  

    PyObject *pReturn = nullptr;
    pReturn = PyEval_CallObject(pFunc, nullptr);
    if(pReturn == nullptr)
    {
        cout<<"Return data is null."<<endl;
        return ;
    }

    char* name=nullptr;
    PyArg_Parse(pReturn, "s", &name);//i表示转换成c++ int型变量
    cout<<"C++: user name is "<<name<<endl;

    Py_DECREF(pModule);
    
}

void call_py_test_class_test(string py_moudle_name)
{
    PyObject * pModule = nullptr;          //Python模块指针，也就是Python文件	
	//导入python文件模块
	pModule = PyImport_ImportModule(py_moudle_name.c_str());
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
	if(pModule == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pModule is null"<<endl;
		return ;
	}
    else
    {
        cout<<"Loading python moudle "<<py_moudle_name<<" OK."<<endl;
    }

    // 直接从模块中获取函数指针
    PyObject *pDict = PyModule_GetDict(pModule);
    if(pDict == nullptr)
    {
        cout << "Line "<<__LINE__<<":test_python pDict is null"<<endl;
        return ;
    }
    else
    {
        cout<<"Loading python dict of moudle "<<py_moudle_name<<" OK."<<endl;
    }

    // 获取类
    PyObject *pClass = PyDict_GetItemString(pDict, "ctest");
    if(pClass == nullptr)
    {
        cout << "Line "<<__LINE__<<":test_python pDict is null"<<endl;
        return ;
    }
    else
    {
        cout<<"Loading python class of moudle "<<py_moudle_name<<" OK."<<endl;
    }

    // 创建类的实例
    PyObject *pInstance = PyInstanceMethod_New(pClass); //python3
    if(pInstance == nullptr)
    {
        cout << "Line "<<__LINE__<<":test_python pInstance is null"<<endl;
        return ;
    }
    else
    {
        cout<<"Create python instance of moudle "<<py_moudle_name<<" OK."<<endl;
    }
    

    // 调用类对象的方法
    // PyObject_CallMethod(pInstance, "increase_score", "(i)", "100");
    PyObject_CallMethod(pInstance, "say_hello", "O",pInstance);
    PyObject *result=nullptr;
    result=PyObject_CallMethod(pInstance, "say", "Os",pInstance,"Test");
    if(result==nullptr)
    {
        cout<<"Call py method failed."<<endl;
    }


    //释放
	Py_DECREF(pInstance);
	Py_DECREF(pClass);
	Py_DECREF(pModule);

}

void call_py_stu_class_test(string py_moudle_name)
{
    PyObject * pModule = nullptr;          //Python模块指针，也就是Python文件	
	//导入python文件模块
	pModule = PyImport_ImportModule(py_moudle_name.c_str());
    // 这里要写调用的python文件，以模块名的形式，而不是以文件的形式
	if(pModule == nullptr)
    {
		cout << "Line "<<__LINE__<<":test_python pModule is null"<<endl;
		return ;
	}
    else
    {
        cout<<"Loading python moudle "<<py_moudle_name<<" OK."<<endl;
    }

    // 直接从模块中获取函数指针
    PyObject *pDict = PyModule_GetDict(pModule);
    if(pDict == nullptr)
    {
        cout << "Line "<<__LINE__<<":test_python pDict is null"<<endl;
        return ;
    }
    else
    {
        cout<<"Loading python dict of moudle "<<py_moudle_name<<" OK."<<endl;
    }

    // 获取类
    PyObject *pClass = PyDict_GetItemString(pDict, "Student");
    if(pClass == nullptr)
    {
        cout << "Line "<<__LINE__<<":test_python pDict is null"<<endl;
        return ;
    }
    else
    {
        cout<<"Loading python class of moudle "<<py_moudle_name<<" OK."<<endl;
    }

    // 创建类的实例
    // PyObject *pInstance = PyInstance_New(pClass,nullptr,nullptr); //python3
    PyObject * pArg=nullptr;

    PyObject *pInstance =  PyEval_CallObject(pClass,pArg);
    if(pInstance == nullptr)
    {
        cout << "Line "<<__LINE__<<":test_python pInstance is null"<<endl;
        return ;
    }
    else
    {
        cout<<"Create python instance of moudle "<<py_moudle_name<<" OK."<<endl;
    }
    

    // 调用类对象的方法
    PyObject_CallMethod(pInstance, "increase_score", "Oi", pInstance, "100");
    PyObject_CallMethod(pInstance, "greet", "O",pInstance);

    PyObject *result=nullptr;
    result=PyObject_CallMethod(pInstance, "say", "Os",pInstance,"Test");
    if(result==nullptr)
    {
        cout<<"Call py method failed."<<endl;
    }


    //释放
	Py_DECREF(pInstance);
	Py_DECREF(pClass);
	Py_DECREF(pModule);

}