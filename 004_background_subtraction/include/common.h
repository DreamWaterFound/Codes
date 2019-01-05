/**
 * @file common.h
 * @author guoqing (1337841346@qq.com)
 * @brief 公共引用头文件
 * @version 0.1
 * @date 2019-01-05
 * 
 * @copyright Copyright (c) 2019
 * 
 */

//头文件引用部分

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>


//名字空间使用说明
using namespace cv;
using namespace std;


/**
 * @brief 数据集的访问路径，这个因人而异
 */
#define DATA_PATH "/home/guoqing/Datasets/CDW-2014/dataset/baseline/highway/input"

/**
 * @brief 帧序列的播放速度
 * 
 */
#define SEQ_FPS 30


