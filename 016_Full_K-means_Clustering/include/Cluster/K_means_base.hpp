#ifndef __K_MEANS_BASE_HPP__
#define __K_MEANS_BASE_HPP__


#include <vector>
#include <cstdlib>
#include <ctime>

//debug
#include <iostream>

template<class SampleType>
class K_Means_Base
{
public:
    K_Means_Base(size_t clusterNum, size_t itNum, double epson, std::vector<SampleType> vSamples);
    ~K_Means_Base();

    //进行计算
    virtual bool Compute(void);

    //获取结果的函数
    inline bool isFailed(void)
    { return mbIsFailed; }

     //获取聚类中心
    inline std::vector<SampleType> GetCenter(void)
    { return mvCenters; }

    //获取样本
    inline std::vector<SampleType> GetSamples(void)
    { return mvSamples; }

    //获取样本的类别
    inline std::vector<size_t> GetSamplesLabel(void)
    { return mvLabels; }

protected:

    // 默认现有的样本点数目远远大于类别的数目,如果不是那么这里就返回false
    virtual bool GetRandomMeans(void);
    // 遍历所有的点,计算到每个类中心的距离,并且对每个点做出所属类别的判断
    virtual void Classified(void);
    // 重新计算每个类别的质心,并且更新; 如果每个类的中心位置变化小于指定的阈值则返回true
    virtual bool ComputeCenter(void);

    //需要自定义实现的

    // 两个样本点的距离描述函数
    virtual double ComputeSamplesDistance(SampleType s1,SampleType s2)=0;
    // 获得一个空样本
    virtual SampleType GetZeroSample(void)=0;
    // 两个样本的相加操作
    virtual SampleType AddSample(SampleType s1, SampleType s2)=0;
    // 样本的取平均操作
    virtual SampleType DevideSample(SampleType s, size_t n)=0;

protected:

    ///聚类个数
    size_t mnClusterNum;
    ///最大迭代次数
    size_t mnItNum;
    ///迭代终止阈值
    double mdEpson;
    ///当前次迭代,聚类中心点的变化差
    double mdErr;
    ///样本集
    std::vector<SampleType> mvSamples;
    ///样本质心
    std::vector<SampleType> mvCenters;
    ///存储每个类别样本标签的向量
    std::vector<int> mvLabels;
    ///存放每次迭代的误差
    std::vector<double> mvdErr;

    ///聚类是否成功 
    bool mbIsFailed;

};


//======================== 下面是功能实现 =====================


template<class SampleType>
K_Means_Base<SampleType>::K_Means_Base(size_t clusterNum, size_t itNum, double epson, std::vector<SampleType> vSamples)
    :mnClusterNum(clusterNum),mnItNum(itNum),mdEpson(epson),mvSamples(vSamples),mbIsFailed(false)
{
    mvLabels=std::vector<int>(mvSamples.size(),-1);
    mvdErr.clear();
}

template<class SampleType>
K_Means_Base<SampleType>::~K_Means_Base()
{
    ;
}

template<class SampleType>
bool K_Means_Base<SampleType>::GetRandomMeans(void)
{
    //由于是base, 所以这里只是实现最基本\最初始的想法

    //检查给出的样本数是否少于要聚类的个数
    if(mvSamples.size()<mnClusterNum) 
    {
        mbIsFailed=true;
        return false;
    }

    //- 1. 准备工作
    size_t n=mvSamples.size();
    std::srand(std::time(NULL));
    //生成对一个样本点是否被使用过的标记
    std::vector<bool> isUsed(n,false);
    mvCenters.clear();

    //- 2. 开始对于每个类,随机选择聚类核心
    for(int i=0;i<mnClusterNum;i++)
    {
        //生成在范围内的随机数
        size_t nCenterID=std::rand()%(n);

        //检查对应的id的样本点是否被使用过
        //对于深度图还要检查其深度值是否有效
        while(isUsed[nCenterID]==true)
        {
            //如果被使用过那么就重新生成
            nCenterID=std::rand()%n;
        }
        

        //如果没有被使用过,就将随机到的这个点作为第i类的类核心
        mvCenters.push_back(mvSamples[nCenterID]);
        //这个点已经被使用了
        isUsed[nCenterID]=true;
    }

    

    return true;
}

template<class SampleType>
void K_Means_Base<SampleType>::Classified(void)
{
    // 准备
    size_t n=mvSamples.size();
    size_t m=mvCenters.size();
    mvLabels.clear();

    // 遍历每个样本点
    for(int i=0;i<n;i++)
    {
        //初始化为到第一个聚类中心的距离
        double min_dis=ComputeSamplesDistance(mvSamples[i],mvCenters[0]);
        size_t min_class=0;

        // 遍历每个类别的中心点
        for(int j=1;j<m;j++)
        {
            // 计算这个样本点到当前遍历到的中心点的距离
            double distance=ComputeSamplesDistance(mvSamples[i],mvCenters[j]);

            // 更新最大值
            if(distance<min_dis)
            {
                min_dis=distance;
                min_class=j;
            }
        }

        // 得到对当前样本点所属类别的判断
        mvLabels.push_back(min_class);
    }
}

template<class SampleType>
bool K_Means_Base<SampleType>::ComputeCenter(void)
{
    //暂时存储每个点的加和;pair的第二个是点的计数
    std::vector<std::pair<SampleType,size_t> > sum;
    sum.clear();

    for(int i=0;i<mvCenters.size();i++)
    {
        //需要自己定义初始化,赋值为0的操作
        SampleType s=GetZeroSample();
        sum.push_back(std::pair<SampleType,size_t>(s,0));
    }

    //开始遍历每个样本
    for(int i=0;i<mvSamples.size();i++)
    {
        sum[mvLabels[i]].first=AddSample(sum[mvLabels[i]].first,mvSamples[i]);
        sum[mvLabels[i]].second++;
    }

    //计算均值
    double err=0.0f;
    for(int i=0;i<mvCenters.size();i++)
    {
        SampleType lastCenter=mvCenters[i];

        //TODO 如果分母为0
        mvCenters[i]=DevideSample(sum[i].first,sum[i].second);

        //计算误差
        err+=ComputeSamplesDistance(mvCenters[i],lastCenter);
    }

    mdErr=err/mvCenters.size();
    
    //完事
    return true;
}

template<class SampleType>
bool K_Means_Base<SampleType>::Compute(void)
{
    // 首先初始化
    if(!GetRandomMeans()) return false;
    mbIsFailed=false;

    // 准备迭代
    for(int i=0;i<mnItNum;i++)
    {
        Classified();
        ComputeCenter();
        mvdErr.push_back(mdErr);
        if(mdErr<mdEpson)   break;
    }
    return true;
}




#endif //__K_MEANS_BASE_HPP__