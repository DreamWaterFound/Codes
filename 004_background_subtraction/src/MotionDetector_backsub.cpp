#include "MotionDetector_backsub.h"

//================构造函数和析构函数====================
//构造函数
MotionDetector_backsub::MotionDetector_backsub()
{
    //复位各种参数
    resetDetector();
    //父类的暂时不考虑
}

//析构函数
MotionDetector_backsub::~MotionDetector_backsub()
{
    ;
}

//================主要功能实现====================
//对当前帧进行运动目标检测
cv::Mat MotionDetector_backsub::motionDetect(cv::Mat frame)
{
    //首先检查一下变量
    if(mpvvContours)
    {
        delete mpvvContours;
    }
    if(mpvRectangles)
    {
        delete mpvRectangles;
    }

    /** 1. 将当前帧图像进行灰度化 */
    cv::cvtColor(frame,mmGrayFrame,CV_BGR2GRAY);

    /** 2. 作差 */
    cv::absdiff(mmGrayBackground,mmGrayFrame,mmDiff);

    /** 3. 阈值化 */
    cv::threshold(mmDiff,mmDiffThresh,
        mnThreshold,MAX_INTENSITY,CV_THRESH_BINARY);

    /** 4. 腐蚀 */
    cv::erode(mmDiffThresh,mmErode,mmKernelErode);

    /** 5. 膨胀 */
    cv::dilate(mmErode,mmDilate,mmKernelDilate);

    /** 6. 寻找并绘制轮廓 */
    mpvvContours = new vector<vector<cv::Point> >;
    cv::findContours(mmDilate,			    //输入的二值图像
				 *mpvvContours,			//轮廓
				 CV_RETR_EXTERNAL,		//仅仅提取最外层的轮廓
				 CV_CHAIN_APPROX_NONE);	//并且存储所有的轮廓点

    mmContours=frame.clone();
    cv::drawContours(mmContours,			//输出图像
				 *mpvvContours,			//要绘制的轮廓
				 -1,					//绘制所有的轮廓
				 mContourColor,		    //颜色
				 mnContourLineWidth);   //线宽

    /** 7. 计算并绘制外接矩形 */
    mpvRectangles=new vector<Rect>(mpvvContours->size());
    mmResult=mmContours.clone();
    //开始遍历
    for(int i=0;i<mpvvContours->size();i++)
    {
        //TODO 可以考虑使用指针的形式
        //计算
        (*mpvRectangles)[i] = cv::boundingRect((*mpvvContours)[i]);
        //绘制
        cv::rectangle(mmResult, 			//输出图像为最终的结果图像
				  (*mpvRectangles)[i], 		//当前遍历到的轮廓的外接矩形
				  mRectColor, 	        //颜色
				  mnRectangleLineWidth);//线宽
    }

    return mmResult;

}

//设置背景模型图像
bool MotionDetector_backsub::setBackground(cv::Mat background)
{
    //首先检查图像是否为空
    if(background.empty())
    {
        return false;
    }

    //设置
    mmBackground=background;
    //并且进行灰度化
    cv::cvtColor(mmBackground,mmGrayBackground,CV_BGR2GRAY);
    
    return true;
}

//================参数设置====================
void MotionDetector_backsub::resetDetector(void)
{
    
    mmBackground=cv::Mat(mFrameSize,CV_8UC3,Scalar(0,0,0));
    mmGrayBackground=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));
    resetDetector_base();
}
