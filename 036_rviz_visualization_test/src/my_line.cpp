/**
 * @file line.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief rviz 绘制线
 * @version 0.1
 * @date 2019-07-24
 * @copyright Copyright (c) 2019
 */

// ROS 支持
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include <cmath>

/**
 * @brief 主函数
 * @param[in] argc 
 * @param[in] argv 
 * @return int 
 */
int main( int argc, char** argv )
{
    //创建一个发布器
    ros::init(argc, argv, "my_line_test");
    ros::NodeHandle n;
    ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("my_line_test", 10);

    // 每隔30绘制？
    ros::Rate r(30);


    while (ros::ok())
    {

        visualization_msgs::Marker line_list;

        //初始化
        line_list.header.frame_id = "/base_link";
        line_list.header.stamp = ros::Time::now();
        line_list.ns = "my_line_test";
        line_list.action = visualization_msgs::Marker::ADD;
        line_list.pose.orientation.w = 1.0;


        //分配3个id
        line_list.id = 0;


        //初始化形状, 这里是演示三种不同的形状
        line_list.type = visualization_msgs::Marker::LINE_LIST;

        //初始化大小, 对应线粗细
        // POINTS markers use x and y scale for width/height respectively
        line_list.scale.x = 0.1;

        //初始化颜色
        // Line list is red
        line_list.color.r = 1.0;
        line_list.color.a = 1.0;

        geometry_msgs::Point p;
        p.x = 1.0;
        p.y = 1.0;
        p.z = 0.0;

        // The line list needs two points for each line
        line_list.points.push_back(p);
        p.y+=50.0;
        p.x+=50.0;
        line_list.points.push_back(p);

        marker_pub.publish(line_list);
        static int cnt=0;

        if(++cnt==30)
        {
            ROS_INFO("Publishing a line.");
            cnt=0;
        }

        r.sleep();

    }
}