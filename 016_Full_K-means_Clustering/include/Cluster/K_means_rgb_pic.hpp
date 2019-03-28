#ifndef __K_MEANS_RGB_PIC_HPP__
#define __K_MEANS_RGB_PIC_HPP__

#include <cmath>

#include "include/Cluster/K_means_base.hpp"
#include "include/types/type_rgb_pixel.hpp"
#include "include/tools/drawData.hpp"

#include <opencv2/opencv.hpp>

class K_Means_RGB : public K_Means_Base<RGB_PIXEL>, public DrawData
{
public:
    K_Means_RGB(size_t clusterNum, size_t itNum, double epson, size_t nHeight, size_t nWitdh, std::vector<RGB_PIXEL> vSamples);
    ~K_Means_RGB();

    bool Compute(void);

    //增加的函数
    void draw(size_t waitTimeMs);

protected:

    // 两个样本点的距离描述函数
    double ComputeSamplesDistance(RGB_PIXEL s1,RGB_PIXEL s2);
    // 获得一个空样本
    RGB_PIXEL GetZeroSample(void);
    // 两个样本的相加操作
    RGB_PIXEL AddSample(RGB_PIXEL s1, RGB_PIXEL s2);
    // 样本的取平均操作
    RGB_PIXEL DevideSample(RGB_PIXEL s, size_t n);

private:
    size_t mnHeight,mnWidth;
    std::vector<cv::Vec3b> mvColor;
};

// ====================== 下面是函数的实现 =========================
K_Means_RGB::K_Means_RGB(size_t clusterNum, size_t itNum, double epson, size_t nHeight, size_t nWitdh, std::vector<RGB_PIXEL> vSamples)
    :K_Means_Base<RGB_PIXEL>(clusterNum,itNum,epson,vSamples),mnHeight(nHeight),mnWidth(nWitdh),DrawData()
{
    //生成颜色配置
    //大概计算一下,TODO  要求clusterNum别太大
    // size_t n=255/clusterNum;
    // for(int i=0;i<clusterNum;i++)
    // {
    //     mvColor.push_back(cv::Vec3b(i*n,i*n,i*n));    
    // }

    size_t n=255*6/clusterNum;
    for(int i=0;i<clusterNum;i++)
    {
        mvColor.push_back(mvHueCircle[i*n]);    
    }
    
}

K_Means_RGB::~K_Means_RGB()
{
    ;
}

double K_Means_RGB::ComputeSamplesDistance(RGB_PIXEL s1,RGB_PIXEL s2)
{
    //暂时只考虑颜色的误差函数
    double dr=s1.r-s2.r;
    double dg=s1.g-s2.g;
    double db=s1.b-s2.b;

    //使用三个差组成向量的均方根作为误差的度量
    return sqrt(dr*dr+dg*dg+db*db);
    
}

RGB_PIXEL K_Means_RGB::GetZeroSample(void)
{
    return RGB_PIXEL(0,0,0,0,0);
}

RGB_PIXEL K_Means_RGB::AddSample(RGB_PIXEL s1, RGB_PIXEL s2)
{
    return RGB_PIXEL(
                s1.u+s2.u,
                s1.v+s2.v,
                s1.r+s2.r,
                s1.g+s2.g,
                s1.b+s2.b
            );
}

RGB_PIXEL K_Means_RGB::DevideSample(RGB_PIXEL s, size_t n)
{
    return RGB_PIXEL(
                s.u/n,
                s.v/n,
                s.r/n,
                s.g/n,
                s.b/n

    );
}

void K_Means_RGB::draw(size_t waitTimeMs)
{
    using namespace std;
    using namespace cv;

    Mat img(mnHeight,mnWidth,CV_8UC3);
    Mat img2(mnHeight,mnWidth,CV_8UC3);

    for(int i=0;i<mnHeight;i++)
    {
        for(int j=0;j<mnWidth;j++)
        {
            //img.at<cv::Vec>(j,i)=mvColor[mvLabels[j+i*mnWidth]];
            RGB_PIXEL p=mvCenters[mvLabels[j+i*mnWidth]];
            img.at<cv::Vec3b>(i,j)=cv::Vec3b(p.b,p.g,p.r);
            img2.at<cv::Vec3b>(i,j)=mvColor[mvLabels[j+i*mnWidth]];
        }
    }

    imshow("res, ",img);
    imshow("res2, ",img2);
    waitKey(waitTimeMs);

}

bool K_Means_RGB::Compute(void)
{
    // 首先初始化


    if(!GetRandomMeans()) return false;

    mbIsFailed=false;
    draw(500);

    // 准备迭代
    for(int i=0;i<mnItNum;i++)
    {
        std::cout<<"it "<<i<<" ..."<<std::endl;
        Classified();
        ComputeCenter();
        draw(500);
        mvdErr.push_back(mdErr);
        std::cout<<"err = "<<mdErr<<std::endl;
        if(mdErr<mdEpson)   break;
    }

    draw(0);

    //mbIsFailed=true;
    return true;
}













#endif //__K_MEANS_RGB_PIC_HPP__