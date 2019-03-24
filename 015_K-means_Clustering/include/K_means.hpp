#ifndef __K_MEANS_HPP__
#define __K_MEANS_HPP__

#include <vector>
#include <cstdlib>
#include <ctime>

#include "Samples.hpp"

class K_MeansCluster
{
public:
    K_MeansCluster(size_t K,size_t N,size_t height, size_t width,double err,std::vector<SampleType> vSamples);
    ~K_MeansCluster();

    //获取结果的函数
    inline bool isFailed(void)
    { return mbIsFailed; }

    //获取聚类中心
    inline std::vector<SampleType> getCenter(void)
    { return mvCenters; }

    //获取样本
    inline std::vector<SampleType> getSamples(void)
    { return mvSamples; }

    void draw(size_t t);
    


private:
    // 默认现有的样本点数目远远大于类别的数目,如果不是那么这里就返回false
    bool getRandomMeans(void);
    // 遍历所有的点,计算到每个类中心的距离,并且对每个点做出所属类别的判断
    void classified(void);
    // 重新计算每个类别的质心,并且更新; 如果每个类的中心位置变化小于指定的阈值则返回true
    bool computeCenter(void);

    // 两个样本点的距离描述函数
    double computeSamplesDistance(SampleType s1,SampleType s2);

    //用于辅助绘图
    size_t mnHeight,mnWidth;

private:
    ///聚类个数
    size_t mnK;
    ///最大迭代次数
    size_t mnN;
    ///迭代终止阈值
    double mdErr;
    ///样本集
    std::vector<SampleType> mvSamples;
    ///样本质心
    std::vector<SampleType> mvCenters;

    ///聚类是否成功 
    bool mbIsFailed;


};

//======================== 下面是功能实现 ==============================

K_MeansCluster::K_MeansCluster(size_t K,size_t N,size_t height, size_t width,double err,std::vector<SampleType> vSamples)
    :mnK(K),mnN(N),mdErr(err),mvSamples(vSamples),mbIsFailed(false),mnHeight(height),mnWidth(width)
{
    //当被构造的时候,执行聚类操作

    //首先是随机选取样本点作为初始聚类的类的中心点
    if(getRandomMeans()==false)
    {
        mbIsFailed=true;
        return ;
    }

    //准备绘制
    draw(0);

    for(int i=0;i<mnN;i++)
    {

        std::cout<<"it "<<i+1<<std::endl;

        classified();

        draw(0);

        computeCenter();

        draw(0);
    }

    std::cout<<"Complete."<<std::endl;
    draw(0);
    

}

K_MeansCluster::~K_MeansCluster()
{
    ;
}


bool K_MeansCluster::getRandomMeans(void)
{
    if(mvSamples.size()<mnK) return false;

    size_t n=mvSamples.size();

    std::srand(std::time(NULL));

    //做标记
    for(auto sample: mvSamples)
    {
        //表示没有被使用过
        sample.cluster_id=-2;
    }

    mvCenters.clear();

    for(int i=0;i<mnK;i++)
    {
        size_t center_id=std::rand()%(n);

        //检查是否之前出现过
        while(mvSamples[center_id].cluster_id==-1)
        {
            //等于-1则说明背使用过,重新生成
            center_id=std::rand()%(n);
        }

        //如果没有被使用过,那么就添加到中心列表中
        mvCenters.push_back(mvSamples[center_id]);
        //打上已经被使用过的标记
        mvSamples[center_id].cluster_id=-1;
    }

    //生成完毕
    return true;
}

void K_MeansCluster::draw(size_t t)
{
    using namespace cv;
    using namespace std;

    Mat img(mnHeight,mnWidth,CV_8UC3,Scalar(200,200,200));

    for(auto sample : mvSamples)
    {
        if(sample.cluster_id==0)
            circle(img,Point(sample.x,sample.y),3,Scalar(0,0,255),-1);
        else if(sample.cluster_id==1)
            circle(img,Point(sample.x,sample.y),3,Scalar(0,255,0),-1);
        else
            circle(img,Point(sample.x,sample.y),3,Scalar(0,0,0),-1);
    }

    size_t size=12;
    for(auto center : mvCenters)
    {
        line(img,Point(center.x-size/2,center.y),Point(center.x+size/2,center.y),Scalar(0,0,0),2);
        line(img,Point(center.x,center.y-size/2),Point(center.x,center.y+size/2),Scalar(0,0,0),2);
    }

    imshow("K-means",img);
    waitKey(t);

}


void K_MeansCluster::classified(void)
{
    // 遍历每个样本点
    size_t n=mvSamples.size();
    size_t m=mvCenters.size();
    for(int i=0;i<n;i++)
    {
        double min_dis=65536;
        size_t min_class;

        //遍历每个类
        for(int j=0;j<m;j++)
        {
            //计算到这个类的中心的距离
            double distance=computeSamplesDistance(mvSamples[i],mvCenters[j]);
            //更新最大值
            if(distance<min_dis)
            {
                //更新
                min_dis=distance;
                min_class=j;
            }         
        }

        //遍历完成了,距离最小的类别将会被判定
        mvSamples[i].cluster_id=min_class;
    }
}

double K_MeansCluster::computeSamplesDistance(SampleType s1,SampleType s2)
{
    //return sqrt((s1.x-s2.x)*(s1.x-s2.x)+(s1.y-s2.y)*(s1.y-s2.y));
    return (double)((s1.x-s2.x)*(s1.x-s2.x)+(s1.y-s2.y)*(s1.y-s2.y));
}

bool K_MeansCluster::computeCenter(void)
{
    using namespace std;

    //暂时存储每个类的点的加和
    vector<SampleType> sum;
    sum.clear();
    for(int i=0;i<mvCenters.size();i++)
    {
        SampleType s(0,0);
        s.cluster_id=0; //在这里这个量用于计数
        sum.push_back(s);
    }
    
    // //初始化sum
    // for(auto s:sum)
    // {
    //     s.x=0;s.y=0;
    //     s.cluster_id=0;
    // }

    //开始遍历每个点
    for(auto sample:mvSamples)
    {
        sum[sample.cluster_id].x+=sample.x;
        sum[sample.cluster_id].y+=sample.y;
        sum[sample.cluster_id].cluster_id++;

    }

    //然后取均值,更新到对应的center中
    for(int i=0;i<mvCenters.size();i++)
    {
        mvCenters[i].x=ceil((sum[i].x*1.0f)/sum[i].cluster_id);
        mvCenters[i].y=ceil((sum[i].y*1.0f)/sum[i].cluster_id);
    }

    //完事,现在先返回true
    return true;

    


    

}



#endif //__K_MEANS_HPP__