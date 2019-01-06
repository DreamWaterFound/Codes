#include <iostream>
#include "opencv2/opencv.hpp"
#include <string>

using namespace cv;
using namespace std;


int dilateSize = 10; ///<膨胀核的大小
int erodeSize = 1;  ///<腐蚀核的大小

//窗口标题
const char*  dilateBar = "dilate_KSize";
const char* dilate_window = "Dilate";
const char* erodeBar = "erode_KSize";
const char* erode_window = "Erode";

//要用到的各种图像
Mat source,             ///<原始图像
    dilate_result,      ///<膨胀结果
    erode_result,       ///<腐蚀结果
    dilate_element,     ///<膨胀核
    erode_element;      ///<腐蚀核

//声明函数原型
void onDilateCallBack(int position,void* userdata);
void onErodeCallBack(int position, void* userdata);


/**
 * @brief 膨胀操作窗口的回调函数
 * @details 猜测这个应该是其中的一种重载形式
 * 
 * @param[in] position 控件滚动条的位置
 * @param[in,out] userdata 虽然说是用户数据，但是实际上传过来的是那个拖动的控制条
 */
void onDilateCallBack(int position,void* userdata) {
    
    //如果控制条的位置，也就要给定的膨胀核的大小不合法，就默认设置为1
    if (position <= 0) {
        position = 1;
    }

    //设置大小，生成核，并且对原始图像进行膨胀处理
    dilateSize = position;
    dilate_element = getStructuringElement(MORPH_RECT, Size(dilateSize, dilateSize));
    dilate(source, dilate_result, dilate_element);
    //HACK
    onErodeCallBack(erodeSize, (void*)erodeBar);
    //显示处理之后的图像
    imshow(dilate_window, dilate_result);
}

/**
 * @brief 腐蚀操作窗口的回调函数
 * 
 * @param[in]     position 控制条的位置
 * @param[in,out] userdata 不清楚用来做什么的用户数据
 */
void onErodeCallBack(int position, void* userdata)
{

    //使得控制条的位置合法化
    if (position <= 0) {
        position = 1;
    }

    //根据控制条的位置设置腐蚀核的大小，并且对图像进行腐蚀操作
    erodeSize = position;
    erode_element = getStructuringElement(MORPH_RECT, Size(erodeSize, erodeSize));
    //erode(source, erode_result, erode_element);
    erode(dilate_result, erode_result, erode_element);
    //显示处理结果
    imshow(erode_window, erode_result);
}



int main(int argc,char* argv[]) 
{
    //检查参数
    if(argc!=2)
    {
        cout<<"[Usage]: "<<argv[0]<<" img_path"<<endl;
        return 0;
    }

    //检查图像是否存在
    source = imread(argv[1], IMREAD_COLOR);
    if(source.empty())
    {
        cout<<"Image is empty! Check image path [ "<<argv[1]<<" ] and checkout the Availability of image."<<endl;
        return 0;
    }

    //显示原始图片
    imshow("Source",source);
    
    //膨胀图像初始值
    //dilate_result = Mat(source.rows,source.cols,CV_8UC3);
    //生成一个opencv窗口
    /*
        brief 创建一个制定的窗口
         
        param[in] name  窗口名称，同时也是窗口的标题
        param[in] flags 窗口的属性标识。可选：
                        - CV_WINDOW_AUTOSIZE 窗口大小会自动调整来适应显示图像
                        - 0 用户可以手动调节窗口大小，不过同事显示的图像的尺寸也会发生变化
        return int 是否创建成功
        note 如果已经存在了这个名字的窗口，那么这个函数将不会做任何事情
         
        int cvNamedWindow( const char* name, int flags=CV_WINDOW_AUTOSIZE );
    */
    cvNamedWindow(dilate_window,        //窗口名称
                  CV_WINDOW_AUTOSIZE);  //窗口自动调整大小
    //并且给该窗口添加一个拖动控制条控件
    //目前从程序调用上来看，这个拖动控制条控件本身就包括了窗口的信息
    /*
        brief 在图像的显示窗口中创建一个滑动空间
        details 用于手动调节阈值，会具有非常直观的效果。
         
        param trackbarname 滑动控件的名称
        param winname 滑动空间所依附得到图像窗口的名称
        param value 初始化值
        param count 滑动控件的刻度范围
        param onChange 回调函数
        param userdata 用户数据，但是一般不用
        return CV_EXPORTS createTrackbar 不知道是什么数据。。。
        
        CV_EXPORTS int createTrackbar(const string& trackbarname, const string& winname,
                              int* value, int count,
                              TrackbarCallback onChange = 0,
                              void* userdata = 0);
    */
    createTrackbar(dilateBar,           //控件名称，这个作为控件的标识，同时也将会被作为提示文本显示在控件上
                   dilate_window,       //所依附的图像窗口的名称
                   &dilateSize,         //初始化值
                   50,                  //刻度范围
                   onDilateCallBack);   //回调函数
    //调用回调函数。正常的话这个应该是opencv的主线程在相关事件发生的时候去调用它的。但是现在我们要手动调用一下。
    onDilateCallBack(dilateSize,                 //控制条的位置
                    (void*)dilateBar);  //控件，含窗口信息。

    //创建腐蚀图像的窗口
    cvNamedWindow(erode_window, CV_WINDOW_AUTOSIZE);
    //创建控件
    createTrackbar(erodeBar, erode_window, &erodeSize, 50, onErodeCallBack);
    //回调
    onErodeCallBack(erodeSize, (void*)erodeBar);
    
    //整个程序的停止条件：当两个窗口中的一个处于激活状态时，按下ESC按键将终止程序
    while(1){ if(waitKey(100)==27)break; } 



    return 0;
}
