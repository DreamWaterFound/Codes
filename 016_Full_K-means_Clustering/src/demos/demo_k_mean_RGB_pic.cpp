#include <iostream>
#include <string>
#include <sstream>

#include "include/types/type_rgb_pixel.hpp"
#include "include/Cluster/K_means_rgb_pic.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main(int argc, char* argv[])
{
    cout<<"K-means on RGB Picture Test."<<endl;

    if(argc!=4)
    {
        cout<<"Usage: "<<argv[0]<<" pic_path class_num iter_max"<<endl;
        cout<<"Ex: "<<argv[0]<<" ./data/img/pic.jpg 2 20"<<endl;
        return 0;
    }

    size_t K,N;

    stringstream ss(argv[2]);
    ss>>K;
    ss=stringstream(argv[3]);
    ss>>N;

    cout<<"K="<<K<<endl;


    Mat srcImg=imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
    if(srcImg.empty())
    {
        cout<<"Empty Img!"<<endl;
        return 0;
    }

    size_t row=srcImg.rows;
    size_t col=srcImg.cols;

    
    vector<RGB_PIXEL> vPixels;
    //构造样本数据
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<col;j++)
        {
            cv::Vec3b color=srcImg.at<cv::Vec3b>(i,j);
            vPixels.push_back(RGB_PIXEL(j,i,
                color[2],color[1],color[0]
            ));
        }
    }

    K_Means_RGB cluster(K,N,0.1,row,col,vPixels);
    cluster.Compute();


    if(cluster.isFailed())
    {
        cout<<"Cluster Failed."<<endl;
        return 0;
    }
    else
    {
        cout<<"Cluster Completed."<<endl;
        return 0;
    }
    
   


}