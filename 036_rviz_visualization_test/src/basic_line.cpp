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
    ros::init(argc, argv, "points_and_lines");
    ros::NodeHandle n;
    ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker_line", 10);

    // 每隔30ms
    ros::Rate r(30);

    // 绘制曲线时,正弦曲线的相位偏移
    float f = 0.0;

    while (ros::ok())
    {

        visualization_msgs::Marker points, line_strip, line_list;

        //初始化
        points.header.frame_id = line_strip.header.frame_id = line_list.header.frame_id = "/base_link";
        points.header.stamp = line_strip.header.stamp = line_list.header.stamp = ros::Time::now();
        points.ns = line_strip.ns = line_list.ns = "points_and_lines";
        points.action = line_strip.action = line_list.action = visualization_msgs::Marker::ADD;
        points.pose.orientation.w = line_strip.pose.orientation.w = line_list.pose.orientation.w = 1.0;


        //分配3个id
        points.id = 0;
        line_strip.id = 1;
        line_list.id = 2;


        //初始化形状
        points.type = visualization_msgs::Marker::POINTS;
        line_strip.type = visualization_msgs::Marker::LINE_STRIP;
        line_list.type = visualization_msgs::Marker::LINE_LIST;

        //初始化大小
        // POINTS markers use x and y scale for width/height respectively
        points.scale.x = 0.2;
        points.scale.y = 0.2;

        // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
        line_strip.scale.x = 0.1;
        line_list.scale.x = 0.1;

        //初始化颜色
        // Points are green
        points.color.g = 1.0f;
        points.color.a = 1.0;

        // Line strip is blue
        line_strip.color.b = 1.0;
        line_strip.color.a = 1.0;

        // Line list is red
        line_list.color.r = 1.0;
        line_list.color.a = 1.0;



        // Create the vertices for the points and lines
        for (uint32_t i = 0; i < 100; ++i)
        {
            float y = 5 * sin(f + i / 100.0f * 2 * M_PI);
            float z = 5 * cos(f + i / 100.0f * 2 * M_PI);

            geometry_msgs::Point p;
            p.x = (int32_t)i - 50;
            p.y = y;
            p.z = z;

            points.points.push_back(p);
            line_strip.points.push_back(p);

            // The line list needs two points for each line
            line_list.points.push_back(p);
            // ?
            p.z += 1.0;
            line_list.points.push_back(p);
        }


        marker_pub.publish(points);
        marker_pub.publish(line_strip);
        marker_pub.publish(line_list);

        r.sleep();

        f += 0.04;
        }
}