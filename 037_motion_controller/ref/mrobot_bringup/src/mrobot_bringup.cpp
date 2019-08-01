#include "mrobot_bringup/mrobot.h"

double RobotV_ = 0;
double YawRate_ = 0;

// 速度控制消息的回调函数
void cmdCallback(const geometry_msgs::Twist& msg)
{
	RobotV_ = msg.linear.x * 1000;
	YawRate_ = msg.angular.z;
}
    
int main(int argc, char** argv)
{
    //初始化ROS节点
	ros::init(argc, argv, "mrobot_bringup");									
    ros::NodeHandle nh;
    
    //初始化MRobot
    // 这种方法，学着 NOTICE
	mrobot::MRobot robot;
    if(!robot.init())
        ROS_ERROR("MRobot initialized failed.");
	ROS_INFO("MRobot initialized successful.");
    
    // 订阅速度topic
    ros::Subscriber sub = nh.subscribe("cmd_vel", 50, cmdCallback);

    //循环运行， 50Hz
    ros::Rate loop_rate(50);
	while (ros::ok()) 
    {
        // 等待接收速度指令 -- //? 提问： 那么如果没有速度被发送，是否就意味着一直卡在这个函数里面出不来了？
        // 不过看样子 spinOnce 也是会遵守这个 loop_rate 的呢
		ros::spinOnce();
        
        // 机器人控制
        robot.spinOnce(RobotV_, YawRate_);
        
		loop_rate.sleep();
	}

	return 0;
}

