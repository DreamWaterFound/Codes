#include <iostream>
#include <string>
#include <sstream>

#include "include/types/type_rgbd_pixel.hpp"
#include "include/Cluster/K-means_rgbd_uv_pic.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main(int argc, char* argv[])
{
    cout<<"K-means on RGB-D Pictures Test."<<endl;

    if(argc!=5)
    {
        cout<<"Usage: "<<argv[0]<<" rgb_pic_path depth_pic_path class_num iter_max"<<endl;
        cout<<"Ex: "<<argv[0]<<" ./data/img/pic.png ./data/img/pic.png 2 20"<<endl;
        return 0;
    }

    size_t K,N;

    stringstream ss(argv[3]);
    ss>>K;
    ss=stringstream(argv[4]);
    ss>>N;

    cout<<"K="<<K<<endl;


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

    //resize(srcImg,srcImg,Size(0,0),0.25,0.25);
    imshow("Origin RGB Image",srcRGBImg);
    imshow("Origin Depth Image",srcDepthImg);

    size_t row=srcRGBImg.rows;
    size_t col=srcRGBImg.cols;

    //对深度图像进行高斯模糊
    Mat srcBlurDepthImg;
    GaussianBlur(srcDepthImg, srcBlurDepthImg, Size(5, 5), 3, 3);
    imshow("Bulr Depth Image",srcBlurDepthImg);

    
    vector<RGBD_PIXEL> vPixels;
    //构造样本数据
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<col;j++)
        {
            if(srcDepthImg.at<uint16_t>(i,j)==0)
                continue ;  //深度值不合法
            else{
            cv::Vec3b color=srcRGBImg.at<cv::Vec3b>(i,j);
            vPixels.push_back(RGBD_PIXEL(j,i,
                color[2],color[1],color[0],
                srcBlurDepthImg.at<uint16_t>(i,j)
            ));
            }
        }
    }

    cout<<"vPixels.size() = "<<vPixels.size()<<endl;
    K_Means_RGBD_UV cluster(K,N,0.1,row,col,vPixels);
    cluster.Compute();

    waitKey(100);

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