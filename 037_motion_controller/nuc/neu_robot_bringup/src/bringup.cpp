#include <iostream>

#include <ros/ros.h>

#include "neu_robot_bringup/neu_robot_bringup.hpp"

using namespace std;

int main(int argc, char* argv[])
{
    ros::init(argc,argv,"neu_robot_bringup");
    ros::NodeHandle nh;
    NEURobot::RobotBringUp Robot(nh);

    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);

    if(Robot.isOK())
    {
        Robot.run();
    }

    return 0;

}
