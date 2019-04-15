#include <iostream>
#include <fstream>
#include <vector>
#include <string>
// #include <assert>

#include <opencv2/opencv.hpp>
#include "TUMRGBD_DataReader.hpp"

using namespace cv;
using namespace std;
using namespace DataReader;

bool getNextPic(TUM_DataReader& reader, cv::Mat& img);

void ExtractAndProcessFeatures(Mat& curImg, Mat& lastImg, vector<Point2f>& vAllPoints, vector<Point2f>& vAllLastPoints, vector<Point2f>& vDynaPoints, vector<Point2f>& vStaticPoints);

void DrawAndSaveResults(Mat& curImg,vector<Point2f>& vDynaPoints, vector<Point2f>& vStaticPoints, string outputPath, size_t id);

int main(int argc, char* argv[])
{
    cout<<"Dynamic Feature Check Demo, Ref DS-SLAM."<<endl;

    if(argc!=4)
    {
        cout<<"Uasge : "<<argv[0]<<" datasets_files_path association_file_path output_path"<<endl;
        cout<<"Ex : "<<argv[0]<<" /home/guoqing/Datasets/TUM_RGBD/fr1/xyz/ /home/guoqing/Datasets/TUM_RGBD/fr1/xyz/associate.txt ./output/"<<endl;
        return 0;
    }

    TUM_DataReader image_reader(argv[1],argv[2]);

    Mat curImg,lastImg;
    vector<Point2f> vAllPoints,vAllLastPoints,vDynaPoints,vStaticPoints;
    size_t id=0;
    // 遍历所有的图片
    while(getNextPic(image_reader,curImg))
    {
        // 首先要进行灰度化
        cvtColor(curImg,curImg,COLOR_BGR2GRAY);
        if(id==0) 
        {
            //提取特征点
            goodFeaturesToTrack(curImg,vAllPoints,1000,0.01,8,cv::Mat(),3,true,0.04);

            // 亚像素级角点位置
            cornerSubPix(curImg,vAllPoints,Size(10,10),Size(-1,-1),TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));

            imshow("res",curImg);
            waitKey(0);


            lastImg=curImg.clone();
            vAllLastPoints=vAllPoints;
            id++;

            continue;
        }

        // 处理
        ExtractAndProcessFeatures(curImg,lastImg, vAllPoints, vAllLastPoints, vDynaPoints, vStaticPoints);

        // 绘制并显示结果
        DrawAndSaveResults(curImg,vDynaPoints,vStaticPoints,argv[3],id);

        lastImg=curImg.clone();
        vAllLastPoints=vAllPoints;
        vAllPoints.clear();
        vDynaPoints.clear();
        vStaticPoints.clear();
        id++;
    }

    return 0;
}


bool getNextPic(TUM_DataReader& reader, cv::Mat& img)
{
    static Mat depth;
    static double timeStamp;
    static vector<double> vGroundTruth;

    return reader.getNextItems(img,depth,timeStamp,vGroundTruth);
}

void ExtractAndProcessFeatures(Mat& curImg, Mat& lastImg, 
                            vector<Point2f>& vAllPoints, vector<Point2f>& vAllLastPoints, 
                            vector<Point2f>& vDynaPoints, vector<Point2f>& vStaticPoints)
{

    // 首先提取当前帧的特征点
    goodFeaturesToTrack(curImg,vAllPoints,1000,0.01,8,cv::Mat(),3,true,0.04);
    cornerSubPix(curImg,vAllPoints,Size(10,10),Size(-1,-1),TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));

    vector<Point2f> vlast=vAllLastPoints;
    vector<Point2f> vcur=vAllPoints;
    // 计算光流金字塔
    std::vector<uchar> state;
    std::vector<float> err;
    assert(!lastImg.empty());
    assert(!curImg.empty());
    assert(vAllLastPoints.size()>0);
    assert(vAllPoints.size()>0);
    // calcOpticalFlowPyrLK(lastImg,curImg,vAllLastPoints,vAllPoints,state,err,Size(22,22),1,TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));
    calcOpticalFlowPyrLK(lastImg,curImg,vlast,vcur,state,err,Size(22,22),1,TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));


    // 点的整理
    vector<Point2f> vpCur,vpLast;

    for(size_t i=0;i<state.size();++i)
    {
        if(state[i])
        {
            vpCur.push_back(vAllPoints[i]);
            vpLast.push_back(vAllLastPoints[i]);
        }
    }


    // 进行粗略的估计
    Mat mask=Mat(Size(1,300),CV_8UC1);
    // 计算基础矩阵
    assert(vpLast.size()>4 && vpCur.size()>4);
    Mat F=findFundamentalMat(vpLast,vpCur,mask,FM_RANSAC,0.1,0.99);

    // 遍历当前帧中的所有特征点，进行几何约束的检查
    for(size_t i=0;i<mask.rows;++i)
    {
        // 如果这个点是内点, 进行极线约束的检查
        if(mask.at<uchar>(i, 0))
        {
            //得到极线参数
            double A = F.at<double>(0, 0)*vpLast[i].x + F.at<double>(0, 1)*vpLast[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0)*vpLast[i].x + F.at<double>(1, 1)*vpLast[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0)*vpLast[i].x + F.at<double>(2, 1)*vpLast[i].y + F.at<double>(2, 2);
            //通过极线约束;来计算误差
            double dd = fabs(A*vpCur[i].x + B*vpCur[i].y + C) / sqrt(A*A + B*B); //Epipolar constraints
            //误差小诶,说明是静态点
            if (dd <= 0.1)
            {
                vStaticPoints.push_back(vpCur[i]);
            }
            else
            {
                vDynaPoints.push_back(vpCur[i]);
            }
            
        }
        else
        {
            // 否则就直接认为是外点
            vDynaPoints.push_back(vpCur[i]);
        }
        
    }

    // TODO 其实这里也可以参考 DS-SLAM 中的操作，进行第二次的极线约束检查

}

void DrawAndSaveResults(Mat& curImg,vector<Point2f>& vDynaPoints, vector<Point2f>& vStaticPoints, string outputPath, size_t id)
{
    Mat res;
    // 绘制结果,首先转换色彩
    cvtColor(curImg,res,COLOR_GRAY2BGR);
    // 绘制静态点, 点的颜色为蓝色
    for(size_t i=0;i<vDynaPoints.size();++i)
    {
        circle(res,vDynaPoints[i],3,Scalar(255,0,0),-1);
    }

    // 绘制动态点，点的颜色为红色
    for(size_t i=0;i<vStaticPoints.size();++i)
    {
        circle(res,vStaticPoints[i],3,Scalar(0,0,255),-1);
    }

    imshow("res",res);
    waitKey(1);

    string id_str("_");
    id_str+=to_string(id);
    id_str+=string(".png");
    
    imwrite(outputPath+id_str,res);
    // imwrite("res.png",res);

    
}