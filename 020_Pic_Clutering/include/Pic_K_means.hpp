#ifndef __PIC_K_MEANS_HPP__
#define __PIC_K_MEANS_HPP__


#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

#include <vector>
#include <cstdlib>
#include <ctime>

#include "tools/drawData.hpp"

#ifndef RAND_MAX
#define RAND_MAX 32767
#endif

#define INVAILD_CLASS 255

using namespace std;

class Pic_K_Means
{
public:
    typedef struct _CenterPoint
    {
        cv::Point2i pt;
        uint16_t d;

        _CenterPoint(cv::Point2i _pt,uint16_t _d):pt(_pt),d(_d){}
    }CenterPoint;

public:
    Pic_K_Means(cv::Mat rgbImg, cv::Mat depthImg, size_t clusterNum, size_t itNum, double epson);
    ~Pic_K_Means();

    bool compute(void);

    //获取结果的函数
    inline bool isFailed(void)
    { return mbIsFailed; }

    void draw(size_t waitTime);

private:
    //得到第一次迭代时的初始聚类
    bool getInitClusters(void);
    // 对所有的像素点进行分类
    void classified(void);
    // 计算每个类的中心
    void computeCenter(void);

    // 计算图像上给出的两个点的图像的距离
    double computePixelsDistance(cv::Point2i p1, cv::Point2i p2, uint16_t d2);

public:
    // 缓存的图片
    cv::Mat mIRGBImg;
    cv::Mat mIDepthImg;

    //压缩格式存储的标签数据
    cv::Mat mILabels;


private:

    ///聚类个数
    size_t mnClusterNum;
    ///最大迭代次数
    size_t mnItNum;
    ///迭代终止阈值
    double mdEpson;
    ///当前次迭代,聚类中心点的变化差
    double mdErr;

    ///样本质心
    // std::vector<CenterPoint> mvCenters;
    std::vector<cv::Point2i> mvCenters;

    std::vector<uint16_t> mvCentersDepth;
    //存放每次迭代的误差
    std::vector<double> mvdErr;

    ///聚类是否成功 
    bool mbIsFailed;

    ///有效像素点的个数
    size_t mnVaild; 

    // 图像信息
    size_t mnRows;
    size_t mnCols;

    std::vector<cv::Vec3b> mvClassColor;

};

// ============================ 下面是功能的实现 =============================
double Pic_K_Means::computePixelsDistance(cv::Point2i p1, cv::Point2i p2, uint16_t d2)
{

    //颜色的误差
    cv::Vec3b color1=mIRGBImg.at<cv::Vec3b>(p1.y,p1.x);
    cv::Vec3b color2=mIRGBImg.at<cv::Vec3b>(p2.y,p2.x);

    double db=color1[0]-color2[0];
    double dg=color1[1]-color2[1];
    double dr=color1[2]-color2[2];
    
    // double dr=s1.r-s2.r;
    // double dg=s1.g-s2.g;
    // double db=s1.b-s2.b;

    double dist_rgb=(dr*dr+dg*dg+db*db);

    //坐标的误差，使用欧式距离
    double dist_pos=(p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y);

    //深度的误差，使用直接的距离
    uint16_t d1=mIDepthImg.at<uint16_t>(p1.y,p1.x);
    // double d2=mIDepthImg.at<uint16_t>(p2.y,p2.x);
    
    double dist_depth=(d2-d1)*(d2-d1);

    //REVIEW 在定义这里的距离度量函数的时候有没有考虑过，可以使用时间上的数据？如果一些点更加倾向于符合或者不符合相机运动，把他们聚类到一期？
    // 以及，在目前的基础上，使用“聚类图像金字塔”，使用上层图像知道下层图像的聚类操作？
    // 类似拓展思想：使用MaskRCNN这种预测特别小的图像（图像金字塔顶层的图像）来提供分类的指导依据（目的是提高实时性），然后在下面几层通过几何的方式逐步恢复聚类效果？

    //此外，让深度梯度变化更大的区域更可能作为聚类的边界，这个怎么做？
    // 以及尝试在彩色图像进行高斯模糊处理？
    // 以及将每个类的mask提取出来，对其进行形态学的腐蚀和膨胀操作？



    //使用三个差组成向量的均方根作为误差的度量
    return sqrt(0.05*dist_rgb+1*dist_pos+1*dist_depth);
}

// ============================ 不怎么会被修改的 =============================

Pic_K_Means::Pic_K_Means(cv::Mat rgbImg, cv::Mat depthImg, size_t clusterNum, size_t itNum, double epson)
    :mIRGBImg(rgbImg.clone()),mIDepthImg(depthImg.clone()),mnClusterNum(clusterNum),mnItNum(itNum),mdEpson(epson),mbIsFailed(false)
{
    //预分配
    mvCenters.reserve(mnClusterNum);

    mnRows=mIRGBImg.rows;
    mnCols=mIRGBImg.cols;

    //255个类,其中255表示深度值不合法或者其他原因,没有被认为是可靠的点
    mILabels=cv::Mat(mnRows,mnCols,CV_8UC1,cv::Scalar(255));

    // 生成颜色
    DrawData ddd(mnClusterNum,mvClassColor);
    std::cout<<"size="<<mvClassColor.size()<<std::endl;
    

}

Pic_K_Means::~Pic_K_Means()
{
    ;
}

bool Pic_K_Means::compute(void)
{
    if(!getInitClusters()) return false;

    for(int i=0;i<mnItNum;++i)
    {
        //std::cout<<"it = "<<i+1<<std::endl;
        classified();
        computeCenter();
       // cout<<"Err = "<<mdErr<<endl;
        if(mdErr<mdEpson) break;
    }
        std::cout<<"Done!"<<std::endl;
    
   
}

//得到第一次迭代时的初始聚类
bool Pic_K_Means::getInitClusters(void)
{
    // 检查给出的样本数是否少于要聚类的个数.这里暂时不考虑那些不可靠的深度点
    size_t n=mnCols*mnRows;
    if(n < mnClusterNum)
    {
        mbIsFailed=true;
        return false;
    }

    mnVaild=0;
    std::srand(std::time(NULL));

    // 像素点是否已经被使用过的标记
    cv::Mat mINotVaild=cv::Mat(mnRows,mnCols,CV_8UC1,cv::Scalar(0));

    // 对于每个类,随机选择聚类中心
    for(int i=0;i<mnClusterNum;++i)
    {
        //产生随机点
        size_t v=std::rand() / (double)(RAND_MAX + 1.0) *mnRows;
        size_t u=std::rand() / (double)(RAND_MAX + 1.0) *mnCols;

        //判断有效性
        while(mINotVaild.at<uint8_t>(v,u) || mIDepthImg.at<uint16_t>(v,u)==0)
        {
            //不行,还得再来一次
            mINotVaild.at<uint8_t>(v,u)=1;

            v=std::rand() / (double)(RAND_MAX + 1.0) *mnRows;
            u=std::rand() / (double)(RAND_MAX + 1.0) *mnCols;            
        }

        // 通过检查,将这个点作为这个类的中心
        mvCenters.emplace_back(u,v);
        mvCentersDepth.push_back(mIDepthImg.at<uint16_t>(v,u));
        //标记这个点已经被使用过了
        mINotVaild.at<uint8_t>(v,u)=1;
    }
    return true;
}

void Pic_K_Means::classified(void)
{
    // 遍历每个有效像素,计算到每个类的距离
    for(int u=0;u<mnCols;++u)
    {
        for(int v=0;v<mnRows;++v)
        {
            // 首先判断这个点的有效性
            //REVIEW 之前写的聚类程序中,这里的处理就不太妥当,最初深度值无效的点,经过高斯滤波后可能反而还有深度值了
            //不过好像当时在处理的时候也注意到了这个问题
            if(mIDepthImg.at<uint16_t>(v,u)==0)
            {
                // 这个点不参与聚类
                //mILabels.at<uint8_t>(v,u)=INVAILD_CLASS;
                continue;
            }
            // 如果这个点有深度值,那么它参与聚类. 
            //初始的距离,为到第一个聚类中心的距离
            
            double min_dis=computePixelsDistance(cv::Point2i(u,v),mvCenters[0],mvCentersDepth[0]);
            size_t min_class=0;

            // 遍历每个类别的中心点
            for(int i=1;i<mnClusterNum;++i)
            {
                // 计算当前的这个像素点到当前遍历到的聚类中心点的距离
                double dist=computePixelsDistance(cv::Point2i(u,v),mvCenters[i],mvCentersDepth[i]);
                // 更新最小距离
                if(min_dis>dist)
                {
                    min_dis=dist;
                    min_class=i;
                }
            }

            // 得到对当前样本点所属类别的判断
            mILabels.at<uint8_t>(v,u)=min_class;
        }
    }
}

void Pic_K_Means::computeCenter(void)
{
    //这里简单地设置为,每个图像块的中心就是其几何中心
    //暂时存储每个点的加和;pair的第二个是点的计数
    std::vector<std::pair<cv::Point2d,size_t> > sum;
    sum.clear();
    std::vector<uint32_t> depth_sum;

    // 首先初始化
    for(int i=0;i<mnClusterNum;++i)
    {
        sum.push_back(std::pair<cv::Point2d,size_t>(cv::Point2d(0.0,0.0),0));
        // sum.emplace_back(cv::Point2d(0.0,0.0),0);
        depth_sum.push_back(0);
    }

    //遍历所有的点
    for(int u=0;u<mnCols;++u)
    {
        for(int v=0;v<mnRows;++v)
        {
            size_t class_id=mILabels.at<uint8_t>(v,u);
            // 判断点是否有效
            if(mIDepthImg.at<uint16_t>(v,u)==0||class_id>=mnClusterNum)
            {
                continue;
            }

            // 累加
            
            sum[class_id].first.x+=u;
            sum[class_id].first.y+=v;
            ++sum[class_id].second;
            depth_sum[class_id]+=mIDepthImg.at<uint16_t>(v,u);
        }
    }

    // 计算均值
    mdErr=0.0f;
    for(int i=0;i<mnClusterNum;++i)
    {
        cv::Point2i last=mvCenters[i];

        mvCenters[i].x=(sum[i].first.x/(double)sum[i].second);
        mvCenters[i].y=(sum[i].first.y/(double)sum[i].second);

       // if(mIDepthImg.at<uint16_t>(mvCenters[i].y,mvCenters[i].x)==0)
        {
            mvCentersDepth[i]=depth_sum[i]/(double)sum[i].second;
        }
        // else
        {
         //   mvCentersDepth[i]=mIDepthImg.at<uint16_t>(mvCenters[i].y,mvCenters[i].x);
        }
    
        //这里就只计算欧式距离了
        mdErr+=sqrt((mvCenters[i].x-last.x)*(mvCenters[i].x-last.x)+(mvCenters[i].y-last.y)*(mvCenters[i].y-last.y));
    }
    mdErr=mdErr/mnClusterNum;

}

void Pic_K_Means::draw(size_t waitTime)
{
    using namespace cv;
    using namespace std;

    Mat img(mnRows,mnCols,CV_8UC3,Scalar(0,0,0));
    Mat mask(mnRows,mnCols,CV_8UC3,Scalar(0,0,0));

    // cout<<"size="<<mvClassColor.size();

    for(int u=0;u<mnCols;++u)
    {
        for(int v=0;v<mnRows;++v)
        {
            if(mIDepthImg.at<uint16_t>(v,u)==0  )
            {
                continue;
            }

            uint8_t classId=mILabels.at<uint8_t>(v,u);

            // if(classId==255) continue;

            // 根据标签来绘制颜色
            
            img.at<cv::Vec3b>(v,u)=mvClassColor[classId];
            mask.at<cv::Vec3b>(v,u)=cv::Vec3b(255,255,255);
        }
    }

    for(int i=0;i<mnClusterNum;++i)
    {
        circle(mask,Point(mvCenters[i].x,mvCenters[i].y),3,Scalar(0,0,255),-1);
    }

    imshow("Result",img);
    imshow("Mask",mask);
    waitKey(waitTime);
    
}





#endif //__PIC_K_MEANS_HPP__
