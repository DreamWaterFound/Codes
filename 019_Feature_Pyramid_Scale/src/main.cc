#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{

    cout<<"Feature Pyramid Scale Test."<<endl;

    if(argc!=2)
    {
        cout<<"Usage: "<<argv[0]<<" img_path"<<endl;
        return 0;
    }

    Mat img=imread(argv[1],CV_LOAD_IMAGE_COLOR);
    if(img.empty()==true)
    {
        cout<<"Error: image "<<argv[1]<<" can not load properly. Please check it."<<endl;
        return 0;
    }

    cv::resize(img,img,cv::Size(640,480));
    cvtColor(img,img,CV_BGR2GRAY);

    std::vector<KeyPoint> keypoints;

    Ptr<ORB> orb=ORB::create(
        500,                    //特征点数目
        1.2f,                   //图像金字塔的缩放倍数
        8,                      //图像金字塔的层数
        31,                     //边界阈值
        0,                      //提取特征点的第一层
        2,                      //?
        ORB::HARRIS_SCORE,      //使用的评分机制
        31,                     //图像patch的大小
        20);	                //fast响应值的阈值
    orb->detect(img,keypoints);

    vector<Scalar> color;
    color.push_back(Scalar(0,255,0));
    color.push_back(Scalar(255,10,255));
    color.push_back(Scalar(0,200,200));
    color.push_back(Scalar(20,255,150));

    color.push_back(Scalar(255,128,200));
    color.push_back(Scalar(0,255,200));
    color.push_back(Scalar(128,128,255));
    color.push_back(Scalar(57,255,255));

    Mat res,res_bk;
    cvtColor(img,res_bk,COLOR_GRAY2BGR);
    res=res_bk.clone();

    vector<string> title;
    title.push_back("res,layer 0");
    title.push_back("res,layer 1");
    title.push_back("res,layer 2");
    title.push_back("res,layer 3");
    title.push_back("res,layer 4");
    title.push_back("res,layer 5");
    title.push_back("res,layer 6");
    title.push_back("res,layer 7");
    


    for(int i=0;i<8;i++)
    {
        for(auto kp: keypoints)
        {
            if(kp.octave==i)
                circle(res,kp.pt,2,color[i],-1);    
        }

        imshow(title[i],res);
        cout<<"layer = "<<i<<endl;
        waitKey(0);

        res=res_bk.clone();
    }

    while(waitKey()!=27);

    return 0;
}