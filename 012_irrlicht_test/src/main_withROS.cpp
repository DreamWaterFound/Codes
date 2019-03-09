/**
 * @file my_subscriber.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 一个简单的图像订阅器
 * @version 0.1
 * @date 2019-03-08
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <irrlicht/irrlicht.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include "drawer.hpp"

using namespace std;



//=========================== 全局变量 =====================




/**
 * @brief 当图像到来时候的回调函数
 * 
 * @param[in] msg 图像信息
 */
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  //其实就是显示图像
  try
  {
    cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::waitKey(10);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}


/**
 * @brief 主函数
 * 
 * @param[in] argc 
 * @param[in] argv 
 * @return int 
 */
int main(int argc, char **argv)
{
    std::cout<<"Started."<<std::endl;
    //ROS操作
    ros::init(argc, argv, "irrlicht_image_listener");    

            

    ros::NodeHandle nh;    

    //opencv创建一个窗口来显示接收到的图像    

    cv::namedWindow("view");    

    cv::startWindowThread();    

    image_transport::ImageTransport it(nh);    

    //这里使用的是ratslam数据集的订阅的图像
    image_transport::Subscriber sub = it.subscribe("irat_red/camera/image", 0, imageCallback);


    Drawer* drawer=new Drawer;
  
    
    
    //控制权w交给ROS

    ros::spin();
    //退出之前记得销毁窗口
    cv::destroyWindow("view");

    delete drawer;

    return 0;
}

