#include "MotionDetector_DiffBase.h"

using namespace std;
using namespace cv;

//================构造函数和析构函数====================
//构造函数
MotionDetector_DiffBase::MotionDetector_DiffBase():
    mpvvContours(nullptr),
    mpvRectangles(nullptr)
{
    //复位各种参数
    //resetDetector();
    //父类的暂时不考虑
}

//析构函数
MotionDetector_DiffBase::~MotionDetector_DiffBase()
{
    //检查并且释放指针
    if(mpvvContours)
    {
        delete mpvvContours;
    }
    if(mpvRectangles)
    {
        delete mpvRectangles;
    }
}

//=============核心功能================
//对当前帧进行运动目标检测
cv::Mat MotionDetector_DiffBase::motionDetect(cv::Mat frame)
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

    /** 1. 得到差分图像 */
    mmDiff=calcuDiffImg(frame);
    
    /** 2. 阈值化 */
    cv::threshold(mmDiff,mmDiffThresh,
        mnThreshold,MAX_INTENSITY,CV_THRESH_BINARY);

    /** 3. 腐蚀 */
    cv::erode(mmDiffThresh,mmErode,mmKernelErode);

    /** 4. 膨胀 */
    cv::dilate(mmErode,mmDilate,mmKernelDilate);

    /** 5. 寻找并绘制轮廓 */
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

    /** 6. 计算并绘制外接矩形 */
    mpvRectangles=new vector<Rect>(mpvvContours->size());
    mmResult=mmContours.clone();
    //开始遍历
    for(int i=0;i<mpvvContours->size();i++)
    {
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

//================参数设置====================
void MotionDetector_DiffBase::resetDetector_base(void)
{
    /** 1. 首先要将所有暂存图像设置为空图像 */
    //mmBackground=cv::Mat(mFrameSize,CV_8UC3,Scalar(0,0,0));
    //mmGrayBackground=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));
    mmGrayFrame=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));
    mmDiff=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));
    mmDiffThresh=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));
    mmErode=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));
    mmDilate=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));
    mmContours=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));
    mmResult=cv::Mat(mFrameSize,CV_8UC1,Scalar(0));

    /** 2. 设置为默认参数 */
    mnThreshold=DEFAULT_THRESHOLD;
    setEropeKernelSize( DEFAULT_EROPE_KERNEL_HEIGHT,
                        DEFAULT_EROPE_KERNEL_WIDTH);
    setDilateKernelSize(DEFAULT_DILATE_KERNEL_HEIGHT,
                        DEFAULT_DILATE_KERNEL_WIDTH);
    mContourColor=DEFAULT_COLOR_RED;
    mRectColor=DEFAULT_COLOR_GREEN;

    mnContourLineWidth=DEFAULT_COMTOUR_LINE_WIDTH;
    mnRectangleLineWidth=DEFAULT_RECTANGLE_LINE_WIDTH;
}

//设置腐蚀核的大小并生成腐蚀核，锚点位置默认
bool MotionDetector_DiffBase::setEropeKernelSize(int heigh, int width)
{
    return setEropeKernelSize(cv::Size(heigh,width));
}

//设置腐蚀核的大小并生成腐蚀核，锚点位置默认
bool MotionDetector_DiffBase::setEropeKernelSize(cv::Size size)
{
    //检查参数
    if(size.height<1 || size.width<1)
    {
        return false;
    }
    else
    {
        mErodeKernelSize=size;
        mmKernelErode=cv::getStructuringElement(MORPH_RECT, mErodeKernelSize);
        return true;
    }
}

//设置膨胀核大小，并且生成部件
bool MotionDetector_DiffBase::setDilateKernelSize(int height,int width)
{
    return setDilateKernelSize(cv::Size(height,width));
}

//设置膨胀核大小，并且生成部件
bool MotionDetector_DiffBase::setDilateKernelSize(cv::Size size)
{
    //检查参数
    if(size.height<1 || size.width<1)
    {
        return false;
    }
    else
    {
        mDilateKernelSize=size;
        mmKernelDilate=cv::getStructuringElement(MORPH_RECT, mDilateKernelSize);
        return true;
    }
}

//设置二值化时的阈值
bool MotionDetector_DiffBase::setBinaryThreshold(int thr)
{
    if(thr>=0 && thr <= 255)
    {
        mnThreshold=thr;
        return true;
    }
    else
    {
        return false;
    }

}

//设置轮廓颜色
bool MotionDetector_DiffBase::setContourColor(int r,int g,int b)
{
    return setContourColor(cv::Scalar(b,g,r));
}

//设置轮廓颜色
bool MotionDetector_DiffBase::setContourColor(cv::Scalar color)
{
    //检查参数
    if(isColor(color))
    {
        mContourColor=color;
        return true;
    }
    else
    {
        return false;
    }
}

//设置轮廓外接矩形颜色
bool MotionDetector_DiffBase::setRectangleColor(int r,int g,int b)
{
    return setRectangleColor(cv::Scalar(b,g,r));
}

//设置轮廓外接矩形颜色
bool MotionDetector_DiffBase::setRectangleColor(cv::Scalar color)
{
     //检查参数
    if(isColor(color))
    {
        mRectColor=color;
        return true;
    }
    else
    {
        return false;
    }
}

//设置轮廓线宽
bool MotionDetector_DiffBase::setContourLineWidth(int width)
{
    if(width>=1)
    {
        mnContourLineWidth=width;
        return true;
    }
    else
    {
        return false;
    }
}

//设置外接矩形线宽
bool MotionDetector_DiffBase::setRectangleLineWidth(int width)
{
    if(width>=1)
    {
        mnRectangleLineWidth=width;
        return true;
    }
    else
    {
        return false;
    }
}

//检查一个cv::Scalar类型的数据是否能够表示一个颜色
bool MotionDetector_DiffBase::isColor(cv::Scalar color)
{
    //第一种情况，单色
    if(
        (color.val[0] >=0 && color.val[0] <=255) &&
        (color.val[1]==0) && (color.val[2]==0) && 
        (color.val[3]==0) 
    )
    {
        return true;
    }

    //第二种情况，三色
    if(
        (color.val[0] >=0 && color.val[0] <=255) &&
        (color.val[1] >=0 && color.val[1] <=255) &&
        (color.val[2] >=0 && color.val[2] <=255) &&
        (color.val[4] == 0) 
    )
    {
        return true;
    }
    else
    {
        return false;
    }
}

