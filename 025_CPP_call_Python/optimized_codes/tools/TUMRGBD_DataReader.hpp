/**
 * @file TUMRGBD_Reader.hpp
 * @author guoqing (1337841346@qq.com)
 * @brief TUM RGBD 数据集读取器的实现
 * @version 0.1
 * @date 2019-02-16
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#ifndef __TUMRGBD_DATA_READER_HPP__
#define __TUMRGBD_DATA_READER_HPP__

#include "DataReader.hpp"
#include <string>
#include <vector>
#include <iostream>

namespace DataReader{

class TUM_DataReader : public DataReaderBase
{
public:
    TUM_DataReader(const std::string datsetsPath, const std::string associationFilePath);
    ~TUM_DataReader();
public:
    bool getItemById(const size_t id, cv::Mat& rgbImg, cv::Mat& depthImg, double& timeStamp);

    bool getRGBImgById(const size_t id, cv::Mat& rgbImg, double& timeStamp);
    bool getDepthImgById(const size_t id, cv::Mat& depthImg, double& timeStamp);

    bool getGroundTruthById(const double id, std::vector<double>& groundTruth, double& timeStamp);

    

    bool getNextItems(cv::Mat& rgbImg, cv::Mat& depthImg, double& timeStamp, std::vector<double>& groundTruth);

public:
    ///数据集路径
    std::string msDatasetPath;
    ///关联文件路径
    std::string msAssociationFilePath;
    
    ///时间戳序列
    std::vector<double> mvfTimeStamps;
    ///彩色图像路径序列
    std::vector<std::string> mvsRGBPath;
    ///深度图像路径序列
    std::vector<std::string> mvsDepthPath;
    ///真值数据序列
    std::vector<std::vector<double>> mvvfGroundTruth;

private:
    
    void resetNextId(void);
    bool checkDataset(void);
private:
    ///下一个要输出的图像id
    size_t mnNextId;
    ///彩色图像大小
    cv::Size mRGBImgSize;
    ///深度图像大小
    cv::Size mDepthImgSize;
    ///彩色图像的通道
    size_t mnRGBImgChannels;
    ///深度图像的通道
    size_t mnDepthImgChannels;

};  //TUM_DataReader


//===================================== 代码实现 ======================================


TUM_DataReader::TUM_DataReader(
    const std::string datsetsPath,
    const std::string associationFilePath):
        msDatasetPath(datsetsPath),
        msAssociationFilePath(associationFilePath)
{
    //定位到最后的目录位置,这个部分的代码直接照搬许可师兄的
    size_t found = msDatasetPath.find_last_of("/\\");
        if(found + 1 != msDatasetPath.size())
            msDatasetPath += msDatasetPath.substr(found, 1);

    //从关联文件中输入数据.其实下面的内容和ORB中的一样,但是多了对真值的读取部分
    std::ifstream file_stream;
    file_stream.open(associationFilePath.c_str());
    while (!file_stream.eof()) {
        std::string s;
        std::getline(file_stream, s);
        if (!s.empty()) {
            std::stringstream ss;
            ss << s;
            double time;
            std::string rgb, depth;
            ss >> time;
            mvfTimeStamps.push_back(time);
            ss >> rgb;
            mvsRGBPath.push_back(datsetsPath + rgb);
            ss >> time;
            ss >> depth;
            mvsDepthPath.push_back(datsetsPath + depth);
            ss >> time;

            //读取真值
            std::vector<double> ground_truth(7);
            for(int i = 0; i < 7; ++i)
                ss >> ground_truth[i];
            mvvfGroundTruth.push_back(ground_truth);
            
        }
    }
    //关闭文件
    file_stream.close();

    resetNextId();

    //图像序列数目的合法性检查
    if(!checkDataset()) mnNumber=0;

    
    
}

bool TUM_DataReader::getItemById(
    const size_t id,
    cv::Mat& rgbImg,
    cv::Mat& depthImg,
    double& timeStamp )
{
    //调用这个函数并不会修改next_id
    if(id>=mnNumber)
    {
        std::cerr << " Index(" << id << ") is out of scape, max should be (0~" << mnNumber - 1 << ")"<<std::endl;
        return false;
    }
    else
    {
        bool res=true;
        double t;
        res&=getRGBImgById(id,rgbImg,t);
        res&=getDepthImgById(id,depthImg,t);
        timeStamp=mvfTimeStamps[id];
        return true;
    }    
}

bool TUM_DataReader::getRGBImgById(
    const size_t id,
    cv::Mat& rgbImg,
    double& timeStamp)
{
    if(id>=mnNumber)
    {
        std::cerr << " Index(" << id << ") is out of scape, max should be (0~" << mnNumber - 1 << ")"<<std::endl;
        return false;
    }
    else
    {
        //准备读取图像
        rgbImg=cv::imread(mvsRGBPath[id],CV_LOAD_IMAGE_UNCHANGED);
        timeStamp=mvfTimeStamps[id];
        if(rgbImg.empty())
        {
            std::cerr<<"RGB image #"<<id<<" is empty!"<<std::endl;
            return false;
        }  
        else
        {
            return true;
        }
    }      
}


bool TUM_DataReader::getDepthImgById(
    const size_t id,
    cv::Mat& depthImg,
    double& timeStamp)
{
    if(id>=mnNumber)
    {
        std::cerr << " Index(" << id << ") is out of scape, max should be (0~" << mnNumber - 1 << ")"<<std::endl;
        return false;
    }
    else
    {
        //准备读取图像
        depthImg=cv::imread(mvsDepthPath[id],CV_LOAD_IMAGE_UNCHANGED);
        timeStamp=mvfTimeStamps[id];
        if(depthImg.empty())
        {
            std::cerr<<"Depth image #"<<id<<" is empty!"<<std::endl;
            return false;
        }  
        else
        {
            return true;
        }
    }
}

bool TUM_DataReader::getGroundTruthById(
    const double id,
    std::vector<double>& groundTruth,
    double& timeStamp)
{
    if(id>=mnNumber)
    {
        std::cerr << " Index(" << id << ") is out of scape, max should be (0~" << mnNumber - 1 << ")"<<std::endl;
        return false;
    }
    else
    {
        //读取数据
        groundTruth=mvvfGroundTruth[id];
        timeStamp=mvfTimeStamps[id];
        return true;
    }
}

bool TUM_DataReader::getNextItems(
    cv::Mat& rgbImg,
    cv::Mat& depthImg,
    double& timeStamp,
    std::vector<double>& groundTruth)
{
    if(mnNextId>=mnNumber)
    {
        std::cerr<<"Image sequence end reached."<<std::endl;
        return false;
    }
    
    bool res=true;
    res&=getRGBImgById(mnNextId,rgbImg,timeStamp);
    res&=getDepthImgById(mnNextId,depthImg,timeStamp);
    res&=getGroundTruthById(mnNextId,groundTruth,timeStamp);    

    mnNextId++;

    return res;
}

void TUM_DataReader::resetNextId(void)
{
    mnNextId=0;
}

bool TUM_DataReader::checkDataset(void)
{
    //首先检查图像序列中的图像数目是否合法
    mnNumber = mvfTimeStamps.size();
    if(mnNumber == 0)
    {
        std::cerr << "No item read! Please check association file: " << msAssociationFilePath << std::endl;
        return false;
    }

    //如果深度图像数目、彩色图像数目和时间戳数目不相等，也要报错
    if(!(mvsDepthPath.size()==mvsRGBPath.size() && mvsRGBPath.size()==mvfTimeStamps.size()))
    {
        std::cerr << "A fatal occured when read item : the numbers of rgb, depth and time stamps are not equal! Please check association file: " << msAssociationFilePath << std::endl;
        return false;
    }

    //使用数据集中的第0个彩色图像进行测试
    cv::Mat img0;
    double timeStamp;
    bool res;
    res=getRGBImgById(0,img0,timeStamp);
    if(!res || img0.empty())
    {
        std::cerr << "Item #0 read fail! Please check the file: " << mvsRGBPath[0] << std::endl;
        return false;
    }
    
    //保存彩色图像的参数
    mRGBImgSize.height=img0.rows;
    mRGBImgSize.width=img0.cols;
    mnRGBImgChannels=img0.channels();

    //接着使用数据即中的第0个深度图像进行测试
    res=getDepthImgById(0,img0,timeStamp);
    if(!res || img0.empty())
    {
        std::cerr << "Item #0 read fail! Please check the file: " << mvsDepthPath[0] << std::endl;
        return false;
    }

    //保存深度图像的参数
    mDepthImgSize.height=img0.rows;
    mDepthImgSize.width=img0.cols;
    mnDepthImgChannels=img0.channels();

    //生成报告
    std::cout<<std::endl;
    std::cout<<"=================== About Selected TUM RGBD Dateset ================="<<std::endl;
    std::cout<<"    Avaiable image items in dataset\t: "<<mnNumber<<std::endl;
    std::cout<<"    RGB image channels\t\t\t: "<<mnRGBImgChannels<<std::endl; 
    std::cout<<"    RGB image size\t\t\t: "<<mRGBImgSize.width<<" x "<<mRGBImgSize.height<<std::endl; 
    std::cout<<"    Depth image channels\t\t: "<<mnDepthImgChannels<<std::endl; 
    std::cout<<"    Depth image size\t\t\t: "<<mDepthImgSize.width<<" x "<<mDepthImgSize.height<<std::endl; 
    std::cout<<"====================================================================="<<std::endl;
    
    return true;
}

TUM_DataReader::~TUM_DataReader()
{
    ;
}

}   //DataReader








#endif //__TUMRGBD_DATA_READER_HPP__