/**
 * @file basic_shape.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 使用rviz绘制mesh模型，参考 
 * @version 0.1
 * @date 2019-07-24
 * 
 * @copyright Copyright (c) 2019
 * 
 */


// ROS 和可视化msg支持
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>           //可视化

/**
 * @brief 主函数
 * @param[in] argc 
 * @param[in] argv 
 * @return int 
 */
int main( int argc, char** argv )
{
  //初始化ROS，幷且创建一个ROS::Publisher 在话题visualization_marker上面
  ros::init(argc, argv, "marker_array_test");
  ros::NodeHandle n;
  ros::Rate r(1);
  ros::Publisher marker_pub = n.advertise<visualization_msgs::MarkerArray>("visualization_markers", 1);

  while (ros::ok())
  {
    //实例化一个Marker消息对象
    visualization_msgs::MarkerArray markers;
    visualization_msgs::Marker marker;

    markers.markers.emplace_back();

    

    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    // 设置frame ID 和 时间戳 
    marker.header.frame_id = "/base_link";
    marker.header.stamp = ros::Time::now();

    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
    // 为这个marker设置一个独一无二的ID，一个marker接收到相同ns和id就会用新的信息代替旧的
    marker.ns = "basic_shapes";
    marker.id = 0;
    

    // 设置marker类型，初始化是立方体。将进行循环
    marker.type = visualization_msgs::Marker::MESH_RESOURCE;

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;

    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
    // 设置marker的位置
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.5 * sqrt(2);
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 0.5 * sqrt(2);

    marker.mesh_resource = "package://rviz_visualization_test/models/tree1.stl";
    // marker.mesh_resource = "package://rviz_visualization_test/models/tree3.stl";
    // /home/guoqing/Codes/036_rviz_visualization_test/models/tree2.stl

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    // 设置marker的大小
    marker.scale.x = 0.01;
    marker.scale.y = 0.01;
    marker.scale.z = 0.01;

    // Set the color -- be sure to set alpha to something non-zero!
    // 设置marker的颜色
    marker.color.r = 118.0 / 256.0f;
    marker.color.g = 182.0 / 256.0f;
    marker.color.b =  16.0 / 256.0f;
    marker.color.a = 1.0f;


    // marker.mesh_use_embedded_materials=true;



    markers.markers.push_back(marker);

    
    // marker.mesh_use_embedded_materials=false;
    marker.id = 1;

    marker.pose.position.x = 2.0f;


    markers.markers.push_back(marker);
    //取消自动删除
    marker.lifetime = ros::Duration();

    // Publish the marker
    // 必须有订阅者才会发布消息
    // NOTE 这里是可以获取订阅者的数目的
    while (marker_pub.getNumSubscribers() < 1)
    {
      if (!ros::ok())
      {
        return 0;
      }
      ROS_WARN_ONCE("Please create a subscriber to the marker");
      sleep(1);
    }

    ROS_INFO("Publishing one marker.");
    marker_pub.publish(markers);


    r.sleep();
  }
}