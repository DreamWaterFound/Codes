/**
 * @file DataReader.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 数据阅读器类的实现
 * @version 0.1
 * @date 2019-01-05
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "common.h"
#include "DataReader.h"


//构造函数
DataReader::DataReader()
{
    //主要工作还是给成员变量初始化
    mnFPS=0;
    mFrameSize.height=0;
    mFrameSize.width=0;
    mnFrameLength=0;
    mnCurFramePos=-1;
}

//析构函数
DataReader::~DataReader()
{
    ;
}

//打开指定路径下的图像序列
bool DataReader::openSeq(const char* path)
{
    //NOTE 所有序列的图片都是从in000001.jpg开始编码的，所以可以通过
    //判断是否存在这个文件，来判断所制定的这个路径是否真的有效
    //不过目前我准备通过文件流的方式来确定
    fstream fs;
    ostringstream ss;

    ss.clear();
    ss<<path<<"/in"<<setw(6)<<setfill('0')<<(int)1<<".jpg";
    fs.open(ss.str(),ios::in);

    if(!fs)
    {
        return false;
    }

    //成功打开，记得关闭文件
    fs.close();
    
    //存档路径
    data_path=path;

    //图像的帧率默认设置为24
    mnFPS=SEQ_FPS;

    //当前帧的位置为0
    mnCurFramePos=0;

    //接下来就是要获取图像帧的大小,首先需要尝试打开第一帧
    checkFrameSizeChannels();

    //确定帧序列的长度
    checkFrameLength();

    return true;
}

//关闭序列
void DataReader::closeSeq(void)
{
    ;
}

//从数据集中读取一个新的帧
bool DataReader::getNewFrame(cv::Mat &img)
{
    //看看是不是当前帧已经是最后一帧了
    if(mnCurFramePos==mnFrameLength)
    {
        //如果是的话就返回false
        return false;
    }

    //现在是说明帧还存在，准备打开
    ostringstream ss;
    ss<<data_path<<"/in"<<setw(6)<<setfill('0')<<++mnCurFramePos<<".jpg";
    cv::Mat res=cv::imread(ss.str(),IMREAD_COLOR);

    //得到的图像为空也不行
    if(res.empty()) return false;
    else  img=res;

    return true;
    
}

//检查图像的第一帧，来确定帧图像的尺寸大小
void DataReader::checkFrameSizeChannels(void)
{
    ostringstream ss;
    ss<<data_path<<"/in"<<setw(6)<<setfill('0')<<(int)1<<".jpg";

    cv::Mat img=imread(ss.str());
    mFrameSize.height=img.rows;
    mFrameSize.width=img.cols;
    mnFrameChannels=img.channels();
}

//确定帧序列的长度
void DataReader::checkFrameLength(void)
{
    int i=1;
    fstream fs;
    ostringstream ss;

    do
    {
        i++;
        //由于打开了，所以现在需要关闭文件
        if(fs)  fs.close();
        //生成下一个文件字符串
        ss.clear();
        ss.str("");
        ss<<data_path<<"/in"<<setw(6)<<setfill('0')<<i<<".jpg";
        fs.open(ss.str(),ios::in);
    }while(fs);

    //此时文件流打开失败，处于关闭状态

    //直到打不开，帧长度就是(i-1)
    mnFrameLength=i-1;

}






