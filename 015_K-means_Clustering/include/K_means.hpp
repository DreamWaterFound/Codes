#ifndef __K_MEANS_HPP__
#define __K_MEANS_HPP__

#include <vector>
#include <cstdlib>
#include <ctime>

#include "Samples.hpp"

class K_MeansCluster
{
public:
    K_MeansCluster(size_t K,size_t N,double err,std::vector<SampleType> vSamples);
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
    


private:
    // 默认现有的样本点数目远远大于类别的数目,如果不是那么这里就返回false
    bool getRandomMeans(void);
    // 遍历所有的点,计算到每个类中心的距离,并且对每个点做出所属类别的判断
    void classified(void);
    // 重新计算每个类别的质心,并且更新; 如果每个类的中心位置变化小于指定的阈值则返回true
    bool computeCenter(void);

    // 两个样本点的距离描述函数
    double computeSamplesDistance(Samples& s1,Samples& s2);

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

K_MeansCluster::K_MeansCluster(size_t K,size_t N,double err,std::vector<SampleType> vSamples)
    :mnK(K),mnN(N),mdErr(err),mvSamples(vSamples),mbIsFailed(false)
{
    //当被构造的时候,执行聚类操作

    //首先是随机选取样本点作为初始聚类的类的中心点
    if(getRandomMeans()==false)
    {
        mbIsFailed=true;
        return ;
    }

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







#endif //__K_MEANS_HPP__