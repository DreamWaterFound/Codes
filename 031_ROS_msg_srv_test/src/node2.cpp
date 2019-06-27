#include "ros/ros.h"
#include "one_pkg/location.h"
#include "one_pkg/cartesian2polar.h"

ros::ServiceClient service_client;
ros::Subscriber    topic_subscriber;

// 给定机器人当前的二维直角坐标系下的坐标(x,y)，判断是否到达某个目标点
// 如果到达该函数返回 true ， 否则返回 false
// 该函数已经在其他文件中实现
bool is_reached(double x,double y);

void check_location_callback(const one_pkg::location::ConstPtr& msg)
{

    ROS_INFO("Get msg.x=%5.2f,msg.y=%5.2f",msg->x,msg->y);

    // if(is_reached(msg->x,msg->y))
    {
        // ROS_INFO("I arrived.");

        one_pkg::cartesian2polar srv;
        srv.request.x=msg->x;
        srv.request.y=msg->y;
        if(service_client.call(srv))
        {
            ROS_INFO("r=%5.2f,theta=%5.2f",srv.response.r,srv.response.theta);
        }
        else
        {
            // ? 检查一下
            ROS_WARN("service cartesian2polar failed.");
        }
    }
}

int main(int argc, char* argv[])
{
    ros::init(argc,argv,"node2");
    ros::NodeHandle n;

    topic_subscriber = n.subscribe("location",10,check_location_callback);
    service_client   = n.serviceClient<one_pkg::cartesian2polar>("cartesian2polar");

    ros::spin();

    return 0;
}

bool is_reached(double x,double y)
{
    double dist=x*x+y*y;
    return dist<1;
}