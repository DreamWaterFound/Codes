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
	

	cout<<"程序开始运行。"<<endl;

	//参数检查,参数0 1-目录 2-模式
	if(argc<3)
	{
		//显示使用信息
		dispUsage(argv[0]);		
		return 0;
	}

	

	//获取参数
	if(argc>2)
	{
		switch(argv[2][0]-'0')
		{
		case 0:
			cout<<"background subtraction method selected."<<endl;
			backSubProc(argv[1]);
			break;
		case 1:
			cout<<"frame subtraction method selected."<<endl;
			frameSubProc(argv[1]);
			break;
		case 2:
			cout<<"GMM method selected."<<endl;
			GMMProc(argv[1]);
			break;
		default:
			cout<<"No match case."<<endl;
			break;
		}
	}
	

    
	return 0;
}

void dispUsage(char *name)
{
	cout<<"[usage] "<<name<<" data_path [mode]"<<endl;
	cout<<"mode:"<<endl;
	cout<<"\t0 - background subtraction"<<endl;
	cout<<"\t1 - frame subtraction"<<endl;
	
}

//使用背景减法来处理
void backSubProc(char* path)
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
	MotionDetector_backsub detector;

	//用于存储当前帧图像
	cv::Mat frame;
	//存储结果的图像
	cv::Mat result;

	if(!(reader.getNewFrame(frame)))
	{
		cout<<"第一帧图像为空。取消。"<<endl;
		return ;
	}
	else
	{
		cout<<"第一帧图像读取成功，正在设置为背景模型"<<endl;
		detector.setBackground(frame);
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
		result=detector.motionDetect(frame);
		
		imshow("result", result);
		

		updateImgs(detector);
		
		if(i==2)
		{
			//输出一些调试信息
			cout<<"第一帧处理完成，暂停。调整好窗口后按任意键，继续."<<endl;
			getchar();
		}

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
		
		/*
		if(i==2)
		{
			//输出一些调试信息
			cout<<"第一帧处理完成，暂停。调整好窗口后按任意键，继续."<<endl;
			getchar();
		}
		*/

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