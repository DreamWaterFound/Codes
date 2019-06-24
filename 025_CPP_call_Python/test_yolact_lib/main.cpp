/**
 * @file main.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 和 lib 协作。
 * @version 0.1
 * @date 2019-06-24
 * 
 * @copyright Copyright (c) 2019
 * 
 */

// =================== 头文件 =========================
#include "yolact_test_v5.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;



// ===================== 正文 =========================
int main(int argc, char* argv[])
{
    cout<<"Test for call yolact to predict image, lib-app Version。"<<endl;
    cout<<"No parameters required."<<endl;
    cout<<"Complied at "<<__TIME__<<" , "<<__DATE__<<" by GNU version "<<__GNUC__<<endl<<endl;


    cout<<"Initlizing Python Environment ..."<<endl;
    init_python_env();
    cout<<"Initlizing Python Environment ... ==> OK."<<endl;

    // =============

    string py_module_name("eval_cpp_interface_v4");
    string py_init_function_name("init_yolact");

    bool res;
    res=init_yolact(py_module_name.c_str(),py_init_function_name.c_str(),
    "/home/guoqing/nets/YOLACT/pre_models/yolact_darknet53_54_800000.pth",
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
    
    string py_eval_function_name("evalimage");    
    if(!transform_image_to_python(img,py_module_name,py_eval_function_name))
    {
        cout<<"call python function \""<<py_eval_function_name<<"\" in python module \""<<py_module_name<<"\" failed."<<endl;
        return false;
    }


    cout<<"Deinitlizing Python Environment ..."<<endl;
    free_python_env();
    cout<<"Deinitlizing Python Environment ... ==> OK."<<endl;


    return 0;
}

