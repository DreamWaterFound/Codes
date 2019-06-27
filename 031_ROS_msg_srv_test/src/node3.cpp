#include "ros/ros.h"
#include "one_pkg/cartesian2polar.h"
#include <cmath>

bool cartesian2polar(one_pkg::cartesian2polar::Request &req,
                     one_pkg::cartesian2polar::Response &res)
{
    res.r=req.x*req.x+req.y*req.y;
    res.theta=atan2(req.y,req.x);

    ROS_INFO("=node3=> Get request: x=%5.2f,y=%5.2f.",req.x,req.y);
    ROS_INFO("=node3=> Sending response: r=%5.2f,theta=%5.2f.",res.r,res.theta);

    return true;
}




int main(int argc, char* argv[])
{
    ros::init(argc,argv,"node3");
    ros::NodeHandle n;

    ros::ServiceServer service = n.advertiseService("cartesian2polar", cartesian2polar);

    ROS_INFO("=node3=> Ready.");

    ros::spin();

    return 0;
}
