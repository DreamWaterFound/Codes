/**
 * @file main.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 主文件
 * @version 0.1
 * @date 2019-01-07
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "main.h"

/**
 * @brief 主函数
 * 
 * @param[in] argc 参数个数
 * @param[in] argv 参数值，是一个字符串
 * @return int 默认为0
 */
int main(int argc,char* argv[])
{
	char *path=argv[1];

	cout<<"程序开始运行。"<<endl;

	//参数检查,参数0 1-目录 2-模式
	if(argc<3)
	{
		//显示使用信息
		dispUsage(argv[0]);		
		return 0;
	}

	// 1.检查参数1_数据集路径

	DataReader reader;

	cout<<"数据库的路径为："<<path<<endl;
	cout<<"正在尝试打开数据库..."<<endl;
	if(reader.openSeq(path))
	{
		//打开成功了
		cout<<"打开成功，数据集的相关信息："<<endl;
		cout<<"大小："<< reader.getFrameSize().height <<" x "<<reader.getFrameSize().width <<endl;
		cout<<"通道："<< reader.getFrameChannels()<<endl;
		cout<<"长度："<< reader.getTotalFrames() <<endl;
		cout<<"FPS: "<< reader.getFPS()<<endl;
	}
	else
	{
		cout<<"\e[1;31m打开失败。\e[0m"<<endl;
		return 0;
	}

	//获取总帧数
	int frameCount = reader.getTotalFrames();
    //获取播放速率
	double FPS = reader.getFPS();
	
	// 2. 检查参数2
	MotionDetector_DiffBase *detector;

	switch(argv[2][0]-'0')
	{
		case 0:
			cout<<"\e[1;33m background subtraction method selected.\e[0m"<<endl;
			detector=new MotionDetector_backsub();
			break;
		case 1:
			cout<<"\e[1;33m frame subtraction method selected.\e[0m"<<endl;
			detector=new MotionDetector_framesub();
			break;
		case 2:
			cout<<"\e[1;33m 3-frame subtraction method selected.\e[0m"<<endl;
			detector=new MotionDetector_3framesub();
			break;
		case 3:
			cout<<"\e[1;33m self-coded GMM method selected.\e[0m"<<endl;
			detector=new MotionDetector_GMM2();
			break;
		case 4:
			cout<<"\e[1;33m opencv GMM method selected.\e[0m"<<endl;
			detector=new MotionDetector_GMM();
			break;
		default:
			cout<<"\e[1;35m No match case. \e[0m"<<endl;
			dispUsage(argv[0]);		
			return 0;
	}


	//准备开始

	//用于存储当前帧图像
	cv::Mat frame;
	//存储结果的图像
	cv::Mat result;

	if(!(reader.getNewFrame(frame)))
	{
		cout<<"\e[1;33m 第一帧图像为空。取消。\e[0m"<<endl;
		return 0;
	}
	
	

	initWindowsPostion(reader.getFrameSize());

    //开始遍历视频序列中的所有帧
	for (int i = 2; i < frameCount; i++)
	{
		if(reader.getNewFrame(frame))
		{
			//如果为真，那么说明读取成功,显示
			cv::imshow("frame",frame);	
	
		}
		else
		{
			//读取不成功？跳过
			continue;
		}

		//进行运动检测
		result=detector->motionDetect(frame);
		
		imshow("result", result);
		

		updateImgs(*detector);
		


		//如果按下了ESC键那么就退出窗口
		if (waitKey(1000.0/FPS) == 27)//按原FPS显示
		{
			cout << "ESC退出!" << endl;
			break;
		}

		//输出进度
		if(i%100==0)
		{
			cout<<"Frame: "<<i<<" / "<<frameCount;
			cout.setf(ios::fixed);
			cout<<setprecision(2)
				<<"\t"<<(100.0*i/(float)frameCount)
				<<" %"<<endl;
			cout<<setprecision(6);
		}
		
	}//对视频中的所有帧进行遍历
	
	cout<<"处理完成。"<<endl;

	

    
	return 0;
}

void dispUsage(char *name)
{
	cout<<"\e[1;32m [usage] "<<name<<" data_path [mode] \e[0m"<<endl;
	cout<<"mode:"<<endl;
	cout<<"\t0 - background subtraction"<<endl;
	cout<<"\t1 - frame subtraction"<<endl;
	cout<<"\t2 - 3-frames subtraction"<<endl;
	cout<<"\t3 - self-code GMM method"<<endl;
	cout<<"\t4 - opencv GMM method"<<endl;
	
}

/*
//使用背景减法来处理
void backSubProc(char* path)
{
	
	
	//设置背景图像
	MotionDetector_backsub detector;

	
}

//使用帧差法来处理
void frameSubProc(char* path)
{
	DataReader reader;

	cout<<"数据库的路径为："<<path<<endl;
	cout<<"正在尝试打开数据库..."<<endl;
	if(reader.openSeq(path))
	{
		//打开成功了
		cout<<"打开成功，数据集的相关信息："<<endl;
		cout<<"大小："<< reader.getFrameSize().height <<" x "<<reader.getFrameSize().width <<endl;
		
		cout<<"通道："<< reader.getFrameChannels()<<endl;
		cout<<"长度："<< reader.getTotalFrames() <<endl;
		cout<<"FPS: "<< reader.getFPS()<<endl;
	}
	else
	{
		cout<<"打开失败。"<<endl;
		return ;
	}

	//获取总帧数
	int frameCount = reader.getTotalFrames();
    //获取播放速率
	double FPS = reader.getFPS();
	
	//设置背景图像
	MotionDetector_3framesub detector;
	//detector.setEropeKernelSize(2,2);

	//用于存储当前帧图像
	cv::Mat frame;
	//存储结果的图像
	cv::Mat result;

	initWindowsPostion(reader.getFrameSize());

    //开始遍历视频序列中的所有帧
	for (int i = 2; i < frameCount; i++)
	{
		if(reader.getNewFrame(frame))
		{
			//如果为真，那么说明读取成功,显示
			cv::imshow("frame",frame);	
			
		}
		else
		{
			//读取不成功？跳过
			continue;
		}

		//进行运动检测
		result=detector.motionDetect(frame);
		
		imshow("result", result);
		
		
		if(i==2)
		{
			//输出一些调试信息
			cout<<"第一帧处理完成，暂停。调整好窗口后按任意键，继续."<<endl;
			getchar();
		}

		updateImgs(detector);

		//如果按下了ESC键那么就退出窗口
		if (waitKey(1000.0/FPS) == 27)//按原FPS显示
		{
			cout << "ESC退出!" << endl;
			break;
		}

		//输出进度
		if(i%100==0)
		{
			cout<<"Frame: "<<i<<" / "<<frameCount;
			cout.setf(ios::fixed);
			cout<<setprecision(2)
				<<"\t"<<(100.0*i/(float)frameCount)
				<<" %"<<endl;
			cout<<setprecision(6);
		}
		
	}//对视频中的所有帧进行遍历
	
	cout<<"处理完成。"<<endl;
}
*/
void initWindowsPostion(cv::Size frameSize)
{
	imshow("frame",cv::Mat(frameSize,CV_8UC3,Scalar(0)));
	cv::moveWindow("frame",50,100);

	imshow("diff",cv::Mat(frameSize,CV_8UC1,Scalar(0)));
	cv::moveWindow("diff",50*2+frameSize.width,100);

	imshow("diff_thr",cv::Mat(frameSize,CV_8UC1,Scalar(0)));
	cv::moveWindow("diff_thr",50*3+frameSize.width*2,100);

	imshow("result",cv::Mat(frameSize,CV_8UC3,Scalar(0)));
	cv::moveWindow("result",50,150+frameSize.height);

	imshow("dilate",cv::Mat(frameSize,CV_8UC1,Scalar(0)));
	cv::moveWindow("dilate",50*2+frameSize.width,150+frameSize.height);

	imshow("erode",cv::Mat(frameSize,CV_8UC1,Scalar(0)));
	cv::moveWindow("erode",50*3+frameSize.width*2,150+frameSize.height);	
}

void updateImgs(MotionDetector_DiffBase &detector)
{
	imshow("diff",detector.getImgDiff());
	imshow("diff_thr",detector.getImgDiffThresh());
	imshow("erode",detector.getImgErode());
	imshow("dilate",detector.getImgDilate());
}


/*
//使用帧差法来处理
void GMMProc(char* path)
{
	DataReader reader;

	cout<<"数据库的路径为："<<path<<endl;
	cout<<"正在尝试打开数据库..."<<endl;
	if(reader.openSeq(path))
	{
		//打开成功了
		cout<<"打开成功，数据集的相关信息："<<endl;
		cout<<"大小："<< reader.getFrameSize().height <<" x "<<reader.getFrameSize().width <<endl;
		
		cout<<"通道："<< reader.getFrameChannels()<<endl;
		cout<<"长度："<< reader.getTotalFrames() <<endl;
		cout<<"FPS: "<< reader.getFPS()<<endl;
	}
	else
	{
		cout<<"打开失败。"<<endl;
		return ;
	}

	//获取总帧数
	int frameCount = reader.getTotalFrames();
    //获取播放速率
	double FPS = reader.getFPS();
	
	//设置背景图像
	MotionDetector_GMM2 detector;
	//detector.setEropeKernelSize(5,5);
	//detector.setBinaryThreshold(220);

	//用于存储当前帧图像
	cv::Mat frame;
	//存储结果的图像
	cv::Mat result;

	initWindowsPostion(reader.getFrameSize());

    //开始遍历视频序列中的所有帧
	for (int i = 2; i < frameCount; i++)
	{
		if(reader.getNewFrame(frame))
		{
			//如果为真，那么说明读取成功,显示
			cv::imshow("frame",frame);	
			
		}
		else
		{
			//读取不成功？跳过
			continue;
		}

		//进行运动检测
		result=detector.motionDetect(frame);
		
		imshow("result", result);
		
		updateImgs(detector);
		


		//如果按下了ESC键那么就退出窗口
		if (waitKey(1000.0/FPS) == 27)//按原FPS显示
		{
			cout << "ESC退出!" << endl;
			break;
		}

		//输出进度
		if(i%100==0)
		{
			cout<<"Frame: "<<i<<" / "<<frameCount;
			cout.setf(ios::fixed);
			cout<<setprecision(2)
				<<"\t"<<(100.0*i/(float)frameCount)
				<<" %"<<endl;
			cout<<setprecision(6);
		}
		
	}//对视频中的所有帧进行遍历
	
	cout<<"处理完成。"<<endl;
}

*/