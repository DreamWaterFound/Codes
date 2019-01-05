
///运动物体检测——背景减法
//包含的这个头文件强啊
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/**
 * @brief 运动物体检测函数
 * 
 * @param background    背景模型
 * @param frame         当前帧
 * @return Mat          结果图像
 */
Mat MoveDetect(Mat background,  
               Mat frame);
 
 /**
  * @brief 主函数
  * 
  * @return int 正常返回
  */
int main()
{
	//声明一个视频捕获类的对象，打开指定的视频文件
	VideoCapture video("bike.avi");//定义VideoCapture类video

    //检查视频文件是否能够被正常打开
	if (!video.isOpened())	//对video进行异常检测
	{
		cout << "video open error!" << endl;
		return 0;
	}

    
	int frameCount = video.get(CV_CAP_PROP_FRAME_COUNT);//获取帧数
	double FPS = video.get(CV_CAP_PROP_FPS);//获取FPS
	Mat frame;//存储帧
	Mat background;//存储背景图像
	Mat result;//存储结果图像
	for (int i = 0; i < frameCount; i++)
	{
		video >> frame;//读帧进frame
		imshow("frame", frame);
		if (frame.empty())//对帧进行异常检测
		{
			cout << "frame is empty!" << endl;
			break;
		}
		int framePosition = video.get(CV_CAP_PROP_POS_FRAMES);//获取帧位置（第几帧）
		cout << "framePosition: " << framePosition << endl;
		if (framePosition == 1)//将第一帧作为背景图像
			background = frame.clone();
		result = MoveDetect(background, frame);//调用MoveDetect()进行运动物体检测，返回值存入result
		imshow("result", result);
		if (waitKey(1000.0/FPS) == 27)//按原FPS显示
		{
			cout << "ESC退出!" << endl;
			break;
		}
	}
	return 0;
}
Mat MoveDetect(Mat background, Mat frame)
{
	Mat result = frame.clone();
	//1.将background和frame转为灰度图
	Mat gray1, gray2;
	cvtColor(background, gray1, CV_BGR2GRAY);
	cvtColor(frame, gray2, CV_BGR2GRAY);
	//2.将background和frame做差
	Mat diff;
	absdiff(gray1, gray2, diff);
	imshow("diff", diff);
	//3.对差值图diff_thresh进行阈值化处理
	Mat diff_thresh;
	threshold(diff, diff_thresh, 50, 255, CV_THRESH_BINARY);
	imshow("diff_thresh", diff_thresh);
	//4.腐蚀
	Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(15, 15));
	erode(diff_thresh, diff_thresh, kernel_erode);
	imshow("erode", diff_thresh);
	//5.膨胀
	dilate(diff_thresh, diff_thresh, kernel_dilate);
	imshow("dilate", diff_thresh);
	//6.查找轮廓并绘制轮廓
	vector<vector<Point>> contours;
	findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓
	//7.查找正外接矩形
	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);
		rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);//在result上绘制正外接矩形
	}
	return result;//返回result
}