#ifndef __SAMPLES_HPP__
#define __SAMPLES_HPP__

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>

//#include <iostream>
#include "type_sample.hpp"

//无法避免的全局变量
bool isUpdated=false;
int ttx,tty;

void onMouse(int event,int x,int y,int flags,void *ustc)
{
    if (event == CV_EVENT_LBUTTONDOWN)//左键按下，读取初始坐标，并在图像上该点处划圆  
    {
        isUpdated=true;
        ttx=x;tty=y;
    }

}

class Samples
{
public:
    Samples(void);

    ~Samples(void);

    // 从图像中获得样本
    std::vector<SampleType> getSamples(size_t height,size_t width);
    // 从文件中获得样本
    std::vector<SampleType> getSamples(std::string fileName);
    // 保存样本到文件
    bool saveSamples(std::string fileName);
    // 查看一下样本
    bool seeSamples(void);

public:
    

private:

    //类中存储的样本
    std::vector<SampleType> mSamples;
    // 是否已经读取过样本
    bool mbIsReaded;

    // 显示样本用图片的大小
    size_t mnHeight,mnWidth;

};

Samples::Samples(void)
    :mbIsReaded(false)
{
       mSamples.clear();
}

Samples::~Samples(void)
{
    ;
}

std::vector<SampleType> Samples::getSamples(size_t height,size_t width)
{
    using namespace std;
    using namespace cv;

    mnHeight= height<10? 10:height;
    mnWidth = width <10? 10:width;

    Mat img(mnHeight,mnWidth,CV_8UC3,cv::Scalar(200,200,200));
    

    imshow("select samples",img);
    setMouseCallback("select samples",onMouse,0);
    
    while(waitKey(1)!=27 && getWindowProperty("select samples",WND_PROP_AUTOSIZE)>=0)
    {
        if(isUpdated)
        {
            isUpdated=false; 
            mSamples.push_back(SampleType(ttx,tty));
            std::cout<<"Add sample ( "<<ttx<<" , "<<tty<<" )"<<std::endl;
            cv::circle(img,cv::Point(ttx,tty),3,cv::Scalar(0,0,255),-1);
            imshow("select samples",img);

        }
    }

    if(!(getWindowProperty("select samples",WND_PROP_AUTOSIZE)<0))
        destroyWindow("select samples");


    if(mSamples.size()>=1)
    {
        mbIsReaded=true;
    }
    
    return mSamples;
    
}

bool Samples::seeSamples(void)
{
    using namespace cv;
    using namespace std;

    if(mbIsReaded==false) return false;

    Mat img(mnHeight,mnWidth,CV_8UC3,Scalar(200,200,200));

    for(auto sample : mSamples)
    {
        circle(img,Point(sample.x,sample.y),3,Scalar(0,0,255),-1);
    }

    imshow("sample points",img);

    waitKey(0);

    return 0;
}

bool Samples::saveSamples(std::string fileName)
{
    using namespace std;

    if(mbIsReaded==false) return false;

    ofstream fs;
    fs.open(fileName.c_str(),ios::out);

    if(!fs) return false;

    fs<<mnHeight<<" "<<mnWidth<<endl;

    for(auto sample : mSamples)
    {
        fs<<sample.x<<" "<<sample.y<<endl;
    }
    fs.flush();
    fs.close();

    return true;
}

std::vector<SampleType> Samples::getSamples(std::string fileName)
{
    using namespace std;

    int tx,ty;

    ifstream fs;
    fs.open(fileName.c_str(),ios::in);

    if(!fs)
    {
        mSamples.clear();
        return mSamples;
    } 

    mSamples.clear();

    fs>>mnHeight>>mnWidth;

    do
    {
        fs>>tx>>ty;
        if(fs.eof()==false)
        {
            mSamples.push_back(SampleType(tx,ty));
        }
    } while (!fs.eof());

    if(mSamples.size()>=1)
        mbIsReaded=true;


    return mSamples;


}








#endif //__SAMPLES_HPP__