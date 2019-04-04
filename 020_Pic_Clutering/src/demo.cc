#include <iostream>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>

#include "Pic_K_means.hpp"

using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
    cout<<"Piture Clustering Demo."<<endl;

    if(argc!=6)
    {
        cout<<"Usage: "<<argv[0]<<" rgb_pic_path depth_pic_path class_num iter_max epson_thr"<<endl;
        cout<<"Ex: "<<argv[0]<<" ./data/img/pic.png ./data/img/pic.png 2 20 1.0"<<endl;
        return 0;
    }

    size_t K,N;
    double epson;

    stringstream ss(argv[3]);
    ss>>K;
    ss=stringstream(argv[4]);
    ss>>N;
    ss=stringstream(argv[5]);
    ss>>epson;

    Mat srcRGBImg=imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
    if(srcRGBImg.empty())
    {
        cout<<"Empty RGB Img!"<<endl;
        return 0;
    }

    Mat srcDepthImg=imread(argv[2],CV_LOAD_IMAGE_UNCHANGED);
    if(srcDepthImg.empty())
    {
        cout<<"Empty Depth Img!"<<endl;
        return 0;
    }

    imshow("Origin RGB Image",srcRGBImg);
    imshow("Origin Depth Image",srcDepthImg);
    // waitKey(0);


    Pic_K_Means Cluster(srcRGBImg,srcDepthImg,K,N,epson);
    Cluster.compute();

    if(Cluster.isFailed())
    {
        cout<<"Cluster Failed."<<endl;
        return 0;
    }
    else
    {
        cout<<"Cluster Completed."<<endl;
        Cluster.draw(0);
        return 0;
    }




    return 0;
}