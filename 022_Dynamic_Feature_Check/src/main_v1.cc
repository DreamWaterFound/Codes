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
typedef pair<Point2f,Point2f> DSPT;
typedef vector<DSPT> DSPTS;

bool getNextPic(TUM_DataReader& reader, cv::Mat& img);

void ExtractAndProcessFeatures(Mat& curImg, Mat& lastImg, vector<Point2f>& vAllPoints, vector<Point2f>& vAllLastPoints, DSPTS& vDynaPoints, DSPTS& vStaticPoints);

void DrawAndSaveResults(Mat& curImg,DSPTS& vDynaPoints, DSPTS& vStaticPoints, string outputPath, size_t id);

int main(int argc, char* argv[])
{
    cout<<"Dynamic Feature Check Demo, Ref DS-SLAM, Version 1.0."<<endl;
    if(argc!=4)
    {
        cout<<"Uasge : "<<argv[0]<<" datasets_files_path association_file_path output_path"<<endl;
        cout<<"Ex : "<<argv[0]<<" /home/guoqing/Datasets/TUM_RGBD/fr1/xyz/ /home/guoqing/Datasets/TUM_RGBD/fr1/xyz/associate.txt ./output/"<<endl;
        return 0;
    }

    TUM_DataReader image_reader(argv[1],argv[2]);

    Mat curImg,lastImg;
    vector<Point2f> vAllPoints,vAllLastPoints;
    DSPTS vDynaPoints,vStaticPoints;
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
                            DSPTS& vDynaPoints, DSPTS& vStaticPoints)
{

    // 首先提取当前帧的特征点
    goodFeaturesToTrack(curImg,vAllPoints,1000,0.01,8,cv::Mat(),3,true,0.04);
    cornerSubPix(curImg,vAllPoints,Size(10,10),Size(-1,-1),TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));

    // ======================= 第一次光流跟踪 =========================

    // 用来避免出现“特征点越来越少”的情况；同时这两个变量记录那些将要使用来进行下一次光流跟踪使用的点
    vector<Point2f> vlast1=vAllLastPoints;
    vector<Point2f> vcur1=vAllPoints;
    // 计算光流金字塔
    std::vector<uchar> state;
    std::vector<float> err;
    calcOpticalFlowPyrLK(lastImg,curImg,vlast1,vcur1,state,err,Size(22,22),1,TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));


    // 点的整理，这里只记录跟踪到的点
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

    // 清空进行下一次计算基础矩阵使用的点
    vector<Point2f> vlast2;
    vector<Point2f> vcur2;
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
                // vStaticPoints.push_back(vpCur[i]);
                // 同时将当前帧的这个点，以及对应的上一帧中的这个点来进行下一步的光流追踪
                vcur2.push_back(vpCur[i]);
                vlast2.push_back(vpLast[i]);
            }
            else
            {
                // vDynaPoints.push_back(vpCur[i]);
            }
            
        }
        else
        {
            // 否则就直接认为是外点
            // vDynaPoints.push_back(vpCur[i]);
        }
        
    }

    // ======================= 第二次光流跟踪 =========================
    // calcOpticalFlowPyrLK(lastImg,curImg,vlast2,vcur2,state,err,Size(22,22),1,TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));
    /*
    vpCur.clear();
    vpLast.clear();
    // 整理跟踪到的点
    for(size_t i=0;i<state.size();++i)
    {
        if(state[i])
        {
            vpCur.push_back(vcur2[i]);
            vpLast.push_back(vlast2[i]);
        }
    }
    */

     // 进行粗略的估计
    mask=Mat(Size(1,300),CV_8UC1);
    // 计算基础矩阵
    assert(vlast2.size()>4 && vcur2.size()>4);
    F=findFundamentalMat(vlast2,vcur2,mask,FM_RANSAC,0.1,0.99);

    // 遍历当前帧中的所有特征点，进行几何约束的检查;参与检查的点是第一次光流跟踪成功的点
    for(size_t i=0;i<vpCur.size();++i)
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
            //REVIEW
            vStaticPoints.push_back(DSPT(vpCur[i],vpLast[i]));
        }
        else
        {
            vDynaPoints.push_back(DSPT(vpCur[i],vpLast[i]));
        }
            
    }
}

void DrawAndSaveResults(Mat& curImg,DSPTS& vDynaPoints, DSPTS& vStaticPoints, string outputPath, size_t id)
{
    Mat res;
    // 绘制结果,首先转换色彩
    cvtColor(curImg,res,COLOR_GRAY2BGR);
    // 绘制动态点, 点的颜色为蓝色
    for(size_t i=0;i<vDynaPoints.size();++i)
    {
        circle(res,vDynaPoints[i].first,3,Scalar(255,0,0),-1);
        // 画线
        // line(res,vDynaPoints[i].first,vDynaPoints[i].second,Scalar(0,255,0),1);
    }

    // 绘制静态点，点的颜色为红色
    for(size_t i=0;i<vStaticPoints.size();++i)
    {
        circle(res,vStaticPoints[i].first,3,Scalar(0,0,255),-1);
        // line(res,vStaticPoints[i].first,vStaticPoints[i].second,Scalar(0,255,0),1);
    }

    imshow("res",res);
    waitKey(1);

    string id_str("_");
    id_str+=to_string(id);
    id_str+=string(".png");
    
    imwrite(outputPath+id_str,res);
    imwrite("res.png",res);

    
}


// TODO
/*  现在的效果不好，估计是算法本身的原因。有个现象是，有的时候动态点会比较少，但是突然有一帧中的动态点就会特别多

    尝试的改进办法： 
    1、（找原因）到底是光流跟踪计算的F不准的问题还是说别的原因？ 尝试将真值中的相机相对位姿变换转换成为基础矩阵，再通过差的二范数的形式来对其进行度量；
    2、（找原因）目前只是绘制出来了当前帧中点的属性问题，为何不尝试一下将上一帧中的点也绘制出来？ 【进行中】
 * 
 * 
 */