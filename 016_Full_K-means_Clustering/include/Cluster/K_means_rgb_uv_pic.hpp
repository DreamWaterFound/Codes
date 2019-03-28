#ifndef __K_MEANS_RGB_UV_PIC_HPP__
#define __K_MEANS_RGB_UV_PIC_HPP__

#include "include/Cluster/K_means_rgb_pic.hpp"

class K_Means_RGB_UV : public K_Means_RGB
{
public:
    K_Means_RGB_UV(size_t clusterNum, size_t itNum, double epson, size_t nHeight, size_t nWitdh, std::vector<RGB_PIXEL> vSamples)
        :K_Means_RGB(clusterNum,itNum,epson,nHeight,nWitdh,vSamples)
    {
        ;   
    }

protected:

    double ComputeSamplesDistance(RGB_PIXEL s1,RGB_PIXEL s2)
    {
        //颜色的误差
        double dr=s1.r-s2.r;
        double dg=s1.g-s2.g;
        double db=s1.b-s2.b;
        double dist_rgb=(dr*dr+dg*dg+db*db);

        //坐标的误差，使用欧式距离
        double dist_pos=(s1.u-s2.u)*(s1.u-s2.u)+(s1.v-s2.v)*(s1.v-s2.v);

        //REVIEW 在定义这里的距离度量函数的时候有没有考虑过，可以使用时间上的数据？如果一些点更加倾向于符合或者不符合相机运动，把他们聚类到一期？
        // 以及，在目前的基础上，使用“聚类图像金字塔”，使用上层图像知道下层图像的聚类操作？
        // 类似拓展思想：使用MaskRCNN这种预测特别小的图像（图像金字塔顶层的图像）来提供分类的指导依据（目的是提高实时性），然后在下面几层通过几何的方式逐步恢复聚类效果？



        //使用三个差组成向量的均方根作为误差的度量
        return sqrt(2*dist_rgb+dist_pos);
    }
};


#endif //__K_MEANS_RGB_UV_PIC_HPP__