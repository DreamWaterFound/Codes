#ifndef __DRAW_DATA_HPP__
#define __DRAW_DATA_HPP__

#include <vector>
#include <opencv2/opencv.hpp>

#include <iostream>
using namespace std;

class DrawData
{
public:
    DrawData(size_t clusterNum,std::vector<cv::Vec3b>& vClassColor)
    {
        //颜色环初始化
        mvHueCircle.reserve(1537);
        mvHueCircle.resize(1537);

        for (int i = 0;i < 255;i++)
        {
            mvHueCircle[i][0] = 255;
            mvHueCircle[i][1] = i;
            mvHueCircle[i][2] = 0;

            mvHueCircle[i+255][0] = 255-i;
            mvHueCircle[i+255][1] = 255;
            mvHueCircle[i+255][2] = 0;

            mvHueCircle[i+511][0] = 0;
            mvHueCircle[i+511][1] = 255;
            mvHueCircle[i+511][2] = i;

            mvHueCircle[i+767][0] = 0;
            mvHueCircle[i+767][1] = 255-i;
            mvHueCircle[i+767][2] = 255;

            mvHueCircle[i+1023][0] = i;
            mvHueCircle[i+1023][1] = 0;
            mvHueCircle[i+1023][2] = 255;

            mvHueCircle[i+1279][0] = 255;
            mvHueCircle[i+1279][1] = 0;
            mvHueCircle[i+1279][2] = 255-i;
        }

        mvHueCircle[1534][0] = 0;
        mvHueCircle[1534][1] = 0;
        mvHueCircle[1534][2] = 0;

        mvHueCircle[1535][0] = 255;
        mvHueCircle[1535][1] = 255;
        mvHueCircle[1535][2] = 255;

        //生成要用到的颜色
        vClassColor.clear();
        size_t n=255*6/clusterNum;
        for(int i=0;i<clusterNum;i++)
        {
            vClassColor.push_back(mvHueCircle[i*n]);    
        }
        
    }



protected:

    std::vector<cv::Vec3b> mvHueCircle;

};




#endif //__DRAW_DATA_HPP__