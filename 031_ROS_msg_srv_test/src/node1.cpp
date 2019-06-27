#include "ros/ros.h"
#include "one_pkg/location.h"


int main(int argc, char* argv[])
{
    ros::init(argc,argv,"node1");
    ros::NodeHandle n;

    ros::Publisher pub = n.advertise<one_pkg::location>("location", 10);
    ros::Rate loop_rate(10);

    while (ros::ok())
    {
        static float cnt=10.0;
        one_pkg::location msg;
        msg.x=cnt;
        msg.y=cnt;   
        ROS_INFO("=node1=> Publishing msg.x=%5.2f,msg.y=%5.2f",msg.x,msg.y);
        pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
        cnt=cnt-0.23;
    }
 
    return 0;
}
