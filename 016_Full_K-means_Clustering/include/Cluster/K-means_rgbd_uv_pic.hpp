#ifndef __K_MEANS_RGB_PIC_HPP__
#define __K_MEANS_RGB_PIC_HPP__

#include <cmath>

#include "include/Cluster/K_means_base.hpp"
#include "include/types/type_rgbd_pixel.hpp"
#include "include/tools/drawData.hpp"

#include <opencv2/opencv.hpp>

class K_Means_RGBD_UV : public K_Means_Base<RGBD_PIXEL>, public DrawData
{
public:
    K_Means_RGBD_UV(size_t clusterNum, size_t itNum, double epson, size_t nHeight, size_t nWitdh, std::vector<RGBD_PIXEL> vSamples);
    ~K_Means_RGBD_UV();

    bool Compute(void);

    //增加的函数
    void draw(size_t waitTimeMs);

protected:

    // 两个样本点的距离描述函数
    virtual double ComputeSamplesDistance(RGBD_PIXEL s1,RGBD_PIXEL s2);
    // 获得一个空样本
    RGBD_PIXEL GetZeroSample(void);
    // 两个样本的相加操作
    RGBD_PIXEL AddSample(RGBD_PIXEL s1, RGBD_PIXEL s2);
    // 样本的取平均操作
    RGBD_PIXEL DevideSample(RGBD_PIXEL s, size_t n);

protected:
    size_t mnHeight,mnWidth;
    std::vector<cv::Vec3b> mvColor;
};

// ====================== 下面是函数的实现 =========================
K_Means_RGBD_UV::K_Means_RGBD_UV(size_t clusterNum, size_t itNum, double epson, size_t nHeight, size_t nWitdh, std::vector<RGBD_PIXEL> vSamples)
    :K_Means_Base<RGBD_PIXEL>(clusterNum,itNum,epson,vSamples),mnHeight(nHeight),mnWidth(nWitdh),DrawData()
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

K_Means_RGBD_UV::~K_Means_RGBD_UV()
{
    ;
}

double K_Means_RGBD_UV::ComputeSamplesDistance(RGBD_PIXEL s1,RGBD_PIXEL s2)
    {

        //颜色的误差
        double dr=s1.r-s2.r;
        double dg=s1.g-s2.g;
        double db=s1.b-s2.b;
        double dist_rgb=(dr*dr+dg*dg+db*db);

        //坐标的误差，使用欧式距离
        double dist_pos=(s1.u-s2.u)*(s1.u-s2.u)+(s1.v-s2.v)*(s1.v-s2.v);

        //深度的误差，使用直接的距离
        double dist_depth=(s1.d-s2.d)*(s1.d-s2.d);

        //REVIEW 在定义这里的距离度量函数的时候有没有考虑过，可以使用时间上的数据？如果一些点更加倾向于符合或者不符合相机运动，把他们聚类到一期？
        // 以及，在目前的基础上，使用“聚类图像金字塔”，使用上层图像知道下层图像的聚类操作？
        // 类似拓展思想：使用MaskRCNN这种预测特别小的图像（图像金字塔顶层的图像）来提供分类的指导依据（目的是提高实时性），然后在下面几层通过几何的方式逐步恢复聚类效果？



        //使用三个差组成向量的均方根作为误差的度量
        return sqrt(dist_rgb+dist_pos+dist_depth);
    }

RGBD_PIXEL K_Means_RGBD_UV::GetZeroSample(void)
{
    return RGBD_PIXEL(0,0,0,0,0,0.0f);
}

RGBD_PIXEL K_Means_RGBD_UV::AddSample(RGBD_PIXEL s1, RGBD_PIXEL s2)
{
    return RGBD_PIXEL(
                s1.u+s2.u,
                s1.v+s2.v,
                s1.r+s2.r,
                s1.g+s2.g,
                s1.b+s2.b,
                s1.d+s2.d
            );
}

RGBD_PIXEL K_Means_RGBD_UV::DevideSample(RGBD_PIXEL s, size_t n)
{
    return RGBD_PIXEL(
                s.u/n,
                s.v/n,
                s.r/n,
                s.g/n,
                s.b/n,
                s.d/n

    );
}

void K_Means_RGBD_UV::draw(size_t waitTimeMs)
{
    using namespace std;
    using namespace cv;

    //创建图像并且先初始化成为黑色
    Mat img(mnHeight,mnWidth,CV_8UC3,Scalar(0,0,0));
    Mat img2(mnHeight,mnWidth,CV_8UC3,Scalar(0,0,0));

    /*
    for(int i=0;i<mnHeight;i++)
    {
        for(int j=0;j<mnWidth;j++)
        {
            //img.at<cv::Vec>(j,i)=mvColor[mvLabels[j+i*mnWidth]];
            RGBD_PIXEL p=mvCenters[mvLabels[j+i*mnWidth]];
            img.at<cv::Vec3b>(i,j)=cv::Vec3b(p.b,p.g,p.r);
            img2.at<cv::Vec3b>(i,j)=mvColor[mvLabels[j+i*mnWidth]];
        }
    }
    */

   for(int i=0;i<mvSamples.size();i++)
   {
       RGBD_PIXEL p=mvCenters[mvLabels[i]];
       uint16_t u=mvSamples[i].u;
       uint16_t v=mvSamples[i].v;
       
        img.at<cv::Vec3b>(v,u)=cv::Vec3b(p.b,p.g,p.r);
    //  img.at<cv::Vec3b>(v,u)=cv::Vec3b(255,255,255);
       img2.at<cv::Vec3b>(v,u)=mvColor[mvLabels[i]];


   }

    imshow("res, ",img);
    imshow("res2, ",img2);
    waitKey(waitTimeMs);

}

bool K_Means_RGBD_UV::Compute(void)
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