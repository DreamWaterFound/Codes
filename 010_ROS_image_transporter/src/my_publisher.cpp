/**
 * @file my_publisher.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 一个简单的图片发布器
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
 

using namespace std;


 /**
  * @brief 主函数
  * 
  * @param[in] argc 
  * @param[in] argv 
  * @return int 
  */
int main(int argc, char** argv)
{
    //参数有效性检查
    if(argc!=2)
    {
        cout<<"Usage: "<<argv[0]<<" image_path"<<endl;
        return -1;
    }

    //节点初始化
    ros::init(argc, argv, "image_publisher");
    ros::NodeHandle nh;
    //注意这里使用的是 image_transport
    image_transport::ImageTransport it(nh);
    //声明发布器,以及它的topic
    image_transport::Publisher pub = it.advertise("mycamera/image", 1);
    //从参数1打开图像文件
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(image.empty())
    {
        cout<<"Image "<<argv[1]<<" is empty!"<<endl;
        ros::shutdown();
        return -1;
    }
    cv::waitKey(30);

    //使用cvbridge进行图像格式的转换
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

    //定义图像的发布速率
    ros::Rate loop_rate(5);
    //只要节点不关闭,那么就发!
    while (nh.ok()) {
        pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}