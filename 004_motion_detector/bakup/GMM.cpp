/**
 * @file GMM.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 参考文件
 * @version 0.1
 * @date 2019-01-08
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>


#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

Mat frame;          ///<当前帧
Mat fgMaskMOG2;     ///<和背景模型有关

Ptr<BackgroundSubtractor> pMOG2;        ///<背景建模类？

//按键
int keyboard; 

/**
 * @brief 对视频进行处理，使用GMM模型找出运动物体
 * 
 * @param videoFilename 视频 路径
 */
void processVideo(string videoFilename) 
{
    // 视频获取
    VideoCapture capture(videoFilename);
    if(!capture.isOpened())
    {
        // 输出视频文件打开错误信息
        cerr << "Unable to open video file: " << videoFilename << endl;
        exit(EXIT_FAILURE);
    }

    // 按下q键和esc退出
    while( (char)keyboard != 'q' && (char)keyboard != 27 )
    {
        // 读取当前帧
        if(!capture.read(frame)) 
        {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }
        // 图像尺寸缩小四分之一
        //目前来看，这个并不是必须的
        cv::resize(frame, frame,cv::Size(), 0.25,0.25);
        //  背景模型生成
        //查一下这个函数的使用方法，GMM模型的使用直接就在这个函数中被解决了
        pMOG2->apply(frame, fgMaskMOG2);

        // 输出当前帧号
        stringstream ss;

        //绘制矩形
        rectangle(frame,                        //目标图像
                  cv::Point(10, 2),             //角点1
                  cv::Point(100,20),            //角点2 
                  cv::Scalar(255,255,255),      //颜色，这里是纯白色
                  -1);                          //填充

        //里面的宏是当前视频帧的位置
        ss << capture.get(CAP_PROP_POS_FRAMES);
        //通过上面的操作，最终得到了一个完整的显示id的字符串
        string frameNumberString = ss.str();
        // 左上角显示帧号
        putText(frame,                  //输出的图像
            frameNumberString.c_str(),  //要绘制的文本
            cv::Point(15, 15),          //字符串在图像上的左下角的坐标
            FONT_HERSHEY_SIMPLEX,       //字体
            0.5 ,                       //在字体指定大小的基础上，字体缩放的系数
            cv::Scalar(0,0,0));         //绘制颜色（白色）
        // 输出结果
        //原始图像
        imshow("Frame", frame); 
        //处理过后的图像
        imshow("FG Mask MOG 2", fgMaskMOG2);
        keyboard = waitKey(30);
    }
    capture.release();
}

/**
 * @brief 主函数
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char* argv[])
{
    // 创建背景建模类
    pMOG2 = createBackgroundSubtractorMOG2(); 
    //视频文件路径
    string inputPath = "F:\\毕业论文相关\\机场视频\\机场.avi";
    //调用自己写的函数
    processVideo(inputPath);
    return 0;
}
