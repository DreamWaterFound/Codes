#include <Python.h>
#include <iostream>
#include <string>
using namespace std;
void init_python_env(){
	Py_Initialize();
	if(!Py_IsInitialized()){
		cout << "initialize failed"<<endl;
	}
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");
//	PyRun_SimpleString("sys.path.append('/home/bob/wkspace/git/detectron/tools/')");
}

void free_python_env(){
	Py_Finalize();
}
//调用输出"Hello Python"函数
int hello()
{
	PyObject * pModule = NULL;
	PyObject * pFunc = NULL;
	PyObject * pDict = NULL;
	//导入python文件模块
	pModule = PyImport_ImportModule("test_python");
	if(pModule == NULL){
		cout << "Line "<<__LINE__<<":test_python pModule is null"<<endl;
		return -1;
	}
	//获取字典属性（可选）
	pDict = PyModule_GetDict(pModule);
	if(pDict == NULL){
		cout << "Line "<<__LINE__<<":test_python pDict is null"<<endl;
		return -1;
	}
	//1.从模块中直接获取函数
//	pFunc = PyObject_GetAttrString(pModule, "hello");//调用的函数名
	//2.从字典属性中获取函数
	pFunc = PyDict_GetItemString(pDict, "hello");
	if(pFunc == NULL){
		cout << "Line "<<__LINE__<<":test_python pFunc is null"<<endl;
		return -1;
	}
    //参数类型转换，传递一个字符串。将c/c++类型的字符串转换为python类型，元组中的python类型查看python文档
	PyRun_SimpleString("print('----------Py_BuildValue')");
	PyObject *pArg = Py_BuildValue("(s)", " c++ bob");
    //调用直接获得的函数，并传递参数
    PyEval_CallObject(pFunc, pArg);
    //PyEval_CallObject(pFunc, NULL);//调用函数,NULL表示参数为空
    Py_DECREF(pModule);
}
//调用Add函数,传两个int型参数
int add()
{
	PyObject * pModule = NULL;
	PyObject * pFunc = NULL;
	pModule = PyImport_ImportModule("test_python");//
	if(pModule == NULL){
		cout << "test_python module add is null"<<endl;
		return -1;
	}
	pFunc = PyObject_GetAttrString(pModule, "add");//Add:Python文件中的函数名
	if(pFunc == NULL){
		cout << "test_python add pFunc is null"<<endl;
		return -1;
	}
	//创建参数:
	PyObject *pArgs = PyTuple_New(2);//函数调用的参数传递均是以元组的形式打包的,2表示参数个数
	PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", 6));//0--序号,i表示创建int型变量
	PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", 8));//1--序号
	// pArg = Py_BuildValue("(i, i)", 6, 8);
	//返回值
	PyObject *pReturn = NULL;
	pReturn = PyEval_CallObject(pFunc, pArgs);//调用函数
	if(pReturn == NULL){
		cout << "test_python add pReturn is null"<<endl;
		return -1;
	}
	//将返回值转换为int类型
	int result;
	PyArg_Parse(pReturn, "i", &result);//i表示转换成c++ int型变量
	cout << "6 + 8 = " << result << endl;
	Py_DECREF(pModule);
}
//调用Add函数,传两个int型参数
int add2(const char* s1,const char*s2)
{
	PyObject * pModule = NULL;
	PyObject * pFunc = NULL;
	pModule = PyImport_ImportModule("test_python");//
	if(pModule == NULL){
		cout << "test_python module add is null"<<endl;
		return -1;
	}
	pFunc = PyObject_GetAttrString(pModule, "add2");//Add:Python文件中的函数名
	if(pFunc == NULL){
		cout << "test_python add pFunc is null"<<endl;
		return -1;
	}
	//创建参数:
	PyObject *pArgs = PyTuple_New(2);//函数调用的参数传递均是以元组的形式打包的,2表示参数个数
	PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", s1));//0--序号,i表示创建int型变量
	PyTuple_SetItem(pArgs, 1, Py_BuildValue("s", s2));//1--序号
	//返回值
	PyObject *pReturn = NULL;
	pReturn = PyEval_CallObject(pFunc, pArgs);//调用函数
	if(pReturn == NULL){
		cout << "test_python add pReturn is null"<<endl;
		return -1;
	}

	char *res;
	PyArg_Parse(pReturn, "s", &res);
	cout << "111s1+s2 = " << res << endl;
	Py_DECREF(pModule);
}
int test_class()
{
	PyObject * pModule = NULL;
	pModule = PyImport_ImportModule("test_python");
	PyObject *pDict = PyModule_GetDict(pModule);
	PyObject *pClass = PyDict_GetItemString(pDict, "ctest");
	PyObject *result =NULL;

	PyRun_SimpleString("print('----------PyInstance_New test class')");
	PyObject *pInstance = PyInstanceMethod_New(pClass); //python3
//  PyObject *pInstance = PyInstance_New(pClass, NULL, NULL);
    if (!pInstance) {
         printf("Cant create second instance./n");
         return -1;
    }
	//调用类的方法
	PyRun_SimpleString("print('----------PyObject_CallMethod')");
	result = PyObject_CallMethod(pInstance, "say", "(s)", "Charity");
	if(result == NULL){
		cout << "test_python class result is null"<<endl;
		return -1;
	}
	//输出返回值
	char* name=NULL;
	PyRun_SimpleString("print('----------PyArg_Parse')");
	PyArg_Parse(result, "s", &name);
	printf("%s\n", name);
	PyRun_SimpleString("print('Python End')");
	//释放
	Py_DECREF(pInstance);
	Py_DECREF(pClass);
	Py_DECREF(pModule);
}
void test_cpp_call_python(){
	init_python_env();
	cout << "调用test_python.py中的Hello函数..." << endl;
	hello();
	cout << "\n调用test_python.py中的Add函数..." << endl;
	add();
	cout << "\n调用test_python.py中的Add2函数..." << endl;
	add2("aaa","bbbb");
	cout << "\n调用test_python.py中的Test类.." << endl;
	test_class();
	free_python_env();
}

int main(int argc, char** argv){
    cout<<"A call python test."<<endl;
    cout<<"Compiled at "<<__TIME__<<__DATE__<<endl<<endl;
	test_cpp_call_python();
    return 0;
}
