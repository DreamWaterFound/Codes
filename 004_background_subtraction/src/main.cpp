
///运动物体检测——背景减法
//包含的这个头文件强啊
#include "common.h"
#include "DataReader.h"
#include "MotionDetector.h"

/**
 * @brief 主函数
 * 
 * @param[in] argc 参数个数
 * @param[in] argv 参数值，是一个字符串
 * @return int 默认为0
 */
int main(int argc,char* argv[])
{
	DataReader reader;

	cout<<"程序开始运行。"<<endl;

	//参数检查,参数0 1-目录 2-模式
	if(argc<2)
	{
		cout<<"[usage] "<<argv[0]<<" data_path [mode]"<<endl;
		return 0;
	}

	cout<<"数据库的路径为："<<argv[1]<<endl;
	cout<<"正在尝试打开数据库..."<<endl;
	if(reader.openSeq(argv[1]))
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
		return 0;
	}

    //获取总帧数
	int frameCount = reader.getTotalFrames();
    //获取播放速率
	double FPS = reader.getFPS();
    
    //用到的一些图像
	cv::Mat frame;//存储帧
	cv::Mat background;//存储背景图像
	cv::Mat result;//存储结果图像

    //开始遍历视频序列中的所有帧
	for (int i = 1; i < frameCount; i++)
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

        //将第一帧作为背景图像        
		if (i == 1)
		{
			background = frame.clone();
		}
			
       
        //调用MoveDetect()进行运动物体检测，返回值存入result，并且显示结果
		result = MotionDetector::back_sub(background, frame);
		
		imshow("result", result);
		

		if(i==1)
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
			cout<<ios::fixed<<setprecision(2)
				<<"\t"<<(100.0*i/(float)frameCount)
				<<" %"<<endl;
			cout<<setprecision(6);
		}
		
	}//对视频中的所有帧进行遍历
	
	cout<<"处理完成。"<<endl;
	return 0;
}

