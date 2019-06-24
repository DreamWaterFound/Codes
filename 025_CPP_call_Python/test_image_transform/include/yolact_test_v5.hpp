#ifndef __YOLACT_HPP__
#define __YOLACT_HPP__

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
bool init_yolact(string py_moudle_name,string py_function_name,
                string trained_model_path,
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

#endif //__YOLACT_HPP__