// 在之前版本的基础上再加上一层优化

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

typedef pair<Point2f,Point2f> Point_Pair;


bool getNextPic(TUM_DataReader& reader, cv::Mat& img);

void ExtractAndProcessFeatures(Mat& curImg, Mat& lastImg, vector<Point2f>& vAllPoints, vector<Point2f>& vAllLastPoints, vector<Point_Pair>& vDynaPoints, vector<Point_Pair>& vStaticPoints);

void DrawAndSaveResults(Mat& curImg,vector<Point_Pair>& vDynaPoints, vector<Point_Pair>& vStaticPoints, string outputPath, size_t id);

int main(int argc, char* argv[])
{
    cout<<"Dynamic Feature Check Demo, Ref DS-SLAM, Version 2.0."<<endl;
    if(argc!=4)
    {
        cout<<"Uasge : "<<argv[0]<<" datasets_files_path association_file_path output_path"<<endl;
        cout<<"Ex : "<<argv[0]<<" /home/guoqing/Datasets/TUM_RGBD/fr1/xyz/ /home/guoqing/Datasets/TUM_RGBD/fr1/xyz/associate.txt ./output/"<<endl;
        return 0;
    }

    cout<<"Press any key to continue ..."<<endl;

    TUM_DataReader image_reader(argv[1],argv[2]);
    cout<<"Press any key to continue ..."<<endl;

    Mat curImg,lastImg;
    vector<Point2f> vAllPoints,vAllLastPoints;
    vector<Point_Pair> vDynaPoints,vStaticPoints,vOutliers;
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
            cout<<"Press any key to continue ..."<<endl;

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
                            vector<Point_Pair>& vDynaPoints, vector<Point_Pair>& vStaticPoints)
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

    const double limit_of_check=2120;
    const int limit_edge_corner=5;

    // 角点质量检查
    for(size_t i=0;i<state.size();++i)
    {
        if(state[i])
        {
            //表示了九宫格点的偏移量
            int dx[10] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
            int dy[10] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
            //获取"匹配"到的角点
            int x1 = vlast1[i].x, y1 = vlast1[i].y;
            int x2 = vcur1[i].x, y2 = vcur1[i].y;

            //判断角点是否落在希望的区域.如果落在了靠近图像边缘的地方,我们就直接扔掉啦
            if ((x1 < limit_edge_corner || x1 >= curImg.cols - limit_edge_corner ||     
                 x2 < limit_edge_corner || x2 >= curImg.cols - limit_edge_corner || 
                 y1 < limit_edge_corner || y1 >= curImg.rows - limit_edge_corner || 
                 y2 < limit_edge_corner || y2 >= curImg.rows - limit_edge_corner))
            {
                //标记为扔掉,并且跳过对这个点的处理
                state[i] = 0;
                continue;
            }

            //对九宫格进行遍历,累加角点周围九宫格像素的差异
            double sum_check = 0;
            for (int j = 0; j < 9; j++)
                sum_check += abs(lastImg.at<unsigned char>(y1 + dy[j], x1 + dx[j]) - curImg.at<unsigned char>(y2 + dy[j], x2 + dx[j]));
            //喏,如果误差太大也不行
            if (sum_check > limit_of_check) state[i] = 0;

            //如果这个点通过了检验,那么就加入到提取得到的当前帧和参考帧的特征点集合中
            if (state[i])
            {
                vpLast.push_back(vlast1[i]);
                vpCur.push_back(vcur1[i]);
            }

            // vpCur.push_back(vAllPoints[i]);
            // vpLast.push_back(vAllLastPoints[i]);
        }
    }


    // 进行粗略的估计
    Mat mask=Mat(Size(1,300),CV_8UC1);
    // 计算基础矩阵
    assert(vpLast.size()>4 && vpCur.size()>4);
    Mat F=findFundamentalMat(vpLast,vpCur,mask,FM_RANSAC,0.1,0.99);

    // 清空进行下一次计算使用的点
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
                // vStaticPoints.push_back(Point_Pair(vpCur[i],vpLast[i]));
                // 同时将当-
                vcur2.push_back(vpCur[i]);
                vlast2.push_back(vpLast[i]);
            }
            else
            {
                // vDynaPoints.push_back(Point_Pair(vpCur[i],vpLast[i]));
            }
        }
        else
        {
            // vDynaPoints.push_back(Point_Pair(vpCur[i],vpLast[i]));
        }
        
        // REVIEW 发现的问题是，在运动模糊出现的时候，这里完全就完蛋了
    }

    // ======================= 第二次计算 =========================
    // 进行粗略的估计
    mask=Mat(Size(1,300),CV_8UC1);
    // 计算基础矩阵
    assert(vpLast.size()>4 && vpCur.size()>4);
    F=findFundamentalMat(vlast2,vcur2,mask,FM_RANSAC,0.1,0.99);

    vector<Point2f> vlast3;
    vector<Point2f> vcur3;
    // 遍历当前帧中的所有特征点，进行几何约束的检查
    for(size_t i=0;i<mask.rows;++i)
    {
        // 如果这个点是内点, 进行极线约束的检查
        if(mask.at<uchar>(i, 0))
        {
            //得到极线参数
            double A = F.at<double>(0, 0)*vlast2[i].x + F.at<double>(0, 1)*vlast2[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0)*vlast2[i].x + F.at<double>(1, 1)*vlast2[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0)*vlast2[i].x + F.at<double>(2, 1)*vlast2[i].y + F.at<double>(2, 2);
            //通过极线约束;来计算误差
            double dd = fabs(A*vcur2[i].x + B*vcur2[i].y + C) / sqrt(A*A + B*B); //Epipolar constraints
            //误差小诶,说明是静态点
            if (dd <= 0.1)
            {
                // vStaticPoints.push_back(Point_Pair(vpCur[i],vpLast[i]));
                // 同时将当-
                vcur3.push_back(vcur2[i]);
                vlast3.push_back(vlast2[i]);
            }
        }
    }

    // ================ 第三次计算 =====================
    // 进行粗略的估计
    mask=Mat(Size(1,300),CV_8UC1);
    // 计算基础矩阵
    assert(vlast3.size()>4 && vcur3.size()>4);
    F=findFundamentalMat(vlast3,vcur3,mask,FM_RANSAC,0.1,0.99);
    // 遍历当前帧中的所有特征点，进行几何约束的检查;参与检查的点是第一次光流跟踪成功的点：vpCur和vpLast
    for(size_t i=0;i<vpCur.size();++i)
    {

        //得到极线参数
        double A = F.at<double>(0, 0)*vpLast[i].x + F.at<double>(0, 1)*vpLast[i].y + F.at<double>(0, 2);
        double B = F.at<double>(1, 0)*vpLast[i].x + F.at<double>(1, 1)*vpLast[i].y + F.at<double>(1, 2);
        double C = F.at<double>(2, 0)*vpLast[i].x + F.at<double>(2, 1)*vpLast[i].y + F.at<double>(2, 2);
        //通过极线约束;来计算误差
        double dd = fabs(A*vpCur[i].x + B*vpCur[i].y + C) / sqrt(A*A + B*B); //Epipolar constraints
        //误差小诶,说明是静态点
        if (dd <= 0.3)
        {
            vStaticPoints.push_back(Point_Pair(vpCur[i],vpLast[i]));
        }
        else
        {
            vDynaPoints.push_back(Point_Pair(vpCur[i],vpLast[i]));
        }
            
    }
    
}

void DrawAndSaveResults(Mat& curImg,vector<Point_Pair>& vDynaPoints, vector<Point_Pair>& vStaticPoints, string outputPath, size_t id)
{
    Mat res;
    // 绘制结果,首先转换色彩
    cvtColor(curImg,res,COLOR_GRAY2BGR);
    // 绘制动态点, 点的颜色为红色
    for(size_t i=0;i<vDynaPoints.size();++i)
    {
        circle(res,vDynaPoints[i].first,3,Scalar(0,0,255),-1);
        // 画线
        line(res,vDynaPoints[i].first,vDynaPoints[i].second,Scalar(0,255,0),1);
    }

    // 绘制静态点，点的颜色为蓝色
    for(size_t i=0;i<vStaticPoints.size();++i)
    {
        circle(res,vStaticPoints[i].first,3,Scalar(255,0,0),-1);
        line(res,vStaticPoints[i].first,vStaticPoints[i].second,Scalar(0,255,0),1);
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
    2、（找原因）目前只是绘制出来了当前帧中点的属性问题，为何不尝试一下将上一帧中的点也绘制出来？ 【完成】
        目前是发现，之前的那些时不时跳出来的“动态点”基本原因都是因为相机运动模糊导致的
 * 
 * 
 */