#ifndef __K_MEANS_2D_POINT_HPP__
#define __K_MEANS_2D_POINT_HPP__

#include  "include/types/type_2D_point.hpp"
#include  "include/Cluster/K_means_base.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>

class K_MeansCluster2DPoint : public K_Means_Base<Points_2D>
{
public:
    K_MeansCluster2DPoint(size_t clusterNum, size_t itNum, double epson, size_t height, size_t width, std::vector<Points_2D> vSamples);

    ~K_MeansCluster2DPoint();

    //重载计算
    bool Compute(void);

    //新增函数: 绘图
    void draw(size_t waitTimeMs);

protected:

    //针对当前样本实现的内容
    //距离描述函数
    double ComputeSamplesDistance(Points_2D s1,Points_2D s2);

    // 获得一个空样本
    Points_2D GetZeroSample(void);

    // 两个样本的相加操作
    Points_2D AddSample(Points_2D s1, Points_2D s2);

    // 样本的取平均操作
    Points_2D DevideSample(Points_2D s, size_t n);

    // 两个样本点的距离描述函数
    //double ComputeSamplesDistance(Points_2D s1,Points_2D s2);

protected:

    size_t mnHeight,mnWidth;

    std::vector<cv::Scalar> mvColor;

};

//======================= 下面是相关功能的实现 ==========================
K_MeansCluster2DPoint::K_MeansCluster2DPoint(size_t clusterNum, size_t itNum, double epson, size_t height, size_t width, std::vector<Points_2D> vSamples)
    :K_Means_Base(clusterNum,itNum,epson,vSamples),mnHeight(height),mnWidth(width)
{
    ;
    mvColor;
}

K_MeansCluster2DPoint::~K_MeansCluster2DPoint()
{
    ;
}

bool K_MeansCluster2DPoint::Compute(void)
{
    using namespace std;
    // 首先初始化
    if(!GetRandomMeans()) return false;
    mbIsFailed=false;
    draw(0);

    // 准备迭代
    for(int i=0;i<mnItNum;i++)
    {
        std::cout<<"it "<<i<<" ... ";
        Classified();
        ComputeCenter();
        std::cout<<"err = "<<mdErr<<std::endl;
        draw(0);
        
        if(mdErr<mdEpson)   break;
    }

    draw(0);
    return true;
}

void K_MeansCluster2DPoint::draw(size_t waitTimeMs)
{
    using namespace cv;
    using namespace std;


    Mat img(mnHeight,mnWidth,CV_8UC3,Scalar(200,200,200));
    //namedWindow("means");

    for(int i=0;i<mvSamples.size();i++)
    {
        
        if(mvLabels[i]==0)
            circle(img,Point(mvSamples[i].x,mvSamples[i].y),3,Scalar(0,0,255),-1);
        else if(mvLabels[i]==1)
            circle(img,Point(mvSamples[i].x,mvSamples[i].y),3,Scalar(0,255,0),-1);
        else if(mvLabels[i]==2)
            circle(img,Point(mvSamples[i].x,mvSamples[i].y),3,Scalar(255,0,255),-1);
        else if(mvLabels[i]==3)
            circle(img,Point(mvSamples[i].x,mvSamples[i].y),3,Scalar(255,255,0),-1);
        else
            circle(img,Point(mvSamples[i].x,mvSamples[i].y),3,Scalar(0,0,0),-1);
        
    }

    size_t size=12;
    for(auto center : mvCenters)
    {
        line(img,Point(center.x-size/2,center.y),Point(center.x+size/2,center.y),Scalar(0,0,0),2);
        line(img,Point(center.x,center.y-size/2),Point(center.x,center.y+size/2),Scalar(0,0,0),2);
    }

    imshow("K-means",img);
    waitKey(waitTimeMs);
}

double K_MeansCluster2DPoint::ComputeSamplesDistance(Points_2D s1,Points_2D s2)
{
    return (double)((s1.x-s2.x)*(s1.x-s2.x)+(s1.y-s2.y)*(s1.y-s2.y));
}

Points_2D K_MeansCluster2DPoint::GetZeroSample(void)
{
    return Points_2D(0,0);
}

Points_2D K_MeansCluster2DPoint::AddSample(Points_2D s1, Points_2D s2)
{
    return Points_2D(s1.x+s2.x,s1.y+s2.y);
}

Points_2D K_MeansCluster2DPoint::DevideSample(Points_2D s, size_t n)
{
    // if(n)
    // {
    //     return Points_2D(s.x*1.0f/n,s.y*1.0f/n);
    // }
    // else
    // {
    //     //其实这种情况的时候就应该算出错了
    //     return Points_2D(0,0);
    // }
            return Points_2D(s.x/n,s.y/n);
}

#endif //__K_MEANS_2D_POINT_HPP__