#ifndef __NEU_ROBOT__
#define __NEU_ROBOT__

#include <ros/ros.h>
#include <ros/time.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <boost/asio.hpp>
#include <geometry_msgs/Twist.h>

// ==== 宏定义 ====
// TODO 变成从配置文件中读取的方式
// 当前节点名称
#define     NODE_NAME               "neu_robot_bringup"
#define     SPEED_TOPIC             "/neu_robot/cmd_vel"
#define     SERIAL_DEVICE           "/dev/ttyUSB0"
#define     LOOP_RATE_HZ            50

namespace NEURobot
{

class RobotBringUp
{
public:
    RobotBringUp(ros::NodeHandle nh);
    ~RobotBringUp();

    // 处理速度的回调函数
    void callback(const geometry_msgs::Twist& msg);

    // run
    void run(void);

    inline bool isOK(void)
    { return is_ok_;}
   
private:

    bool initSerialPort(void);
    // 接收速度，发布里程计，发布坐标变换
    bool spinOnce(void);
    
    // 读取速度（轮式里程计数值）
    bool readSpeed();
    // 写入速度
    bool writeSpeed(double linear_speed_x, double angle_speed_yaw);

    // 初始化数据结构
    void initDataBuf(void);
   
private:
    ros::Time current_time_, last_time_;

    bool is_ok_;

    double x_;
    double y_;
    double th_;

    double vx_;
    double vy_;
    double vth_;

    // 来自消息的速度值，这里我们只关心这两个
    double linear_speed_x_;
    double angle_speed_yaw_;

    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;

    tf::TransformBroadcaster odom_broadcaster_;

    boost::asio::io_service iosev_;
    boost::asio::serial_port* p_sp;

    uint8_t robot_speed_cmd_[15];

};

RobotBringUp::RobotBringUp(ros::NodeHandle nh)
    : x_(0),    y_(0),  th_(0),
      vx_(0),   vy_(0), vth_(0),
      nh_(nh),  is_ok_(true),
      linear_speed_x_(0.0f),angle_speed_yaw_(0.0f)
{
    initDataBuf();


    if(!initSerialPort())
    {
        ROS_FATAL_STREAM("One Error occured when operating serial decive \""<<SERIAL_DEVICE<<"\".");
        is_ok_=false;
    }
    else
    {
        ROS_INFO("NEU_Robot initialized successful.");
    
        // 订阅速度topic
        sub_ = nh_.subscribe(SPEED_TOPIC, 50, &RobotBringUp::callback, this);
        // FIXME:
        // pub_ = nh_.advertise<nav_msgs::Odometry>("odom", 50);	
    }
}

RobotBringUp::~RobotBringUp()
{
    ;
}

// 速度消息的回调函数
void RobotBringUp::callback(const geometry_msgs::Twist& msg)
{
	linear_speed_x_  = msg.linear.x * 1000;
	angle_speed_yaw_ = msg.angular.z;
}

bool RobotBringUp::initSerialPort(void)
{
    try
    {
        p_sp= new boost::asio::serial_port(iosev_, SERIAL_DEVICE);

        // 串口连接. 这个参数是电机控制器固定死的，所以这里就写死在程序里面了
        p_sp->set_option(boost::asio::serial_port::baud_rate(115200));
        p_sp->set_option(boost::asio::serial_port::flow_control(boost::asio::serial_port::flow_control::none));
        p_sp->set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none));
        p_sp->set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one));
        p_sp->set_option(boost::asio::serial_port::character_size(8));
    }
    catch (...)
    {
        return false;
    }
    
    ros::Time::init();
	current_time_   = ros::Time::now();
	last_time_      = ros::Time::now();
    return true;
}



bool RobotBringUp::spinOnce(void)
{
    // 这边相当用函数的参数做了一个缓冲
    writeSpeed(linear_speed_x_,angle_speed_yaw_);

    return true;
}

bool RobotBringUp::readSpeed()
{
   
    // FIXME:


    return true;
}

bool RobotBringUp::writeSpeed(double linear_speed_x, double angle_speed_yaw)
{
    // 符号判断
    robot_speed_cmd_[ 3]= ( (linear_speed_x  >= 0) ? '-':' ' );
    robot_speed_cmd_[ 9]= ( (angle_speed_yaw >= 0) ? ' ':'-' );
    linear_speed_x  = fabs(linear_speed_x);
    angle_speed_yaw = fabs(angle_speed_yaw);


    // 限幅
    uint16_t linear = linear_speed_x   > 1000? (uint16_t)1000 : (uint16_t)linear_speed_x  ;
    uint16_t angle  = angle_speed_yaw  > 1000? (uint16_t)1000 : (uint16_t)angle_speed_yaw ;

    // 还是先用比较笨的办法把速度分离出来吧
    robot_speed_cmd_[ 4] = fabs(linear_speed_x )  > 1000? '1':'0';
    robot_speed_cmd_[ 5] = (linear / 100) % 10 + '0';
    robot_speed_cmd_[ 6] = (linear / 10 ) % 10 + '0';
    robot_speed_cmd_[ 7] = (linear      ) % 10 + '0';

    robot_speed_cmd_[10] = fabs(angle_speed_yaw)  > 1000? '1':'0';
    robot_speed_cmd_[11] = (angle / 100) % 10 + '0';
    robot_speed_cmd_[12] = (angle / 10 ) % 10 + '0';
    robot_speed_cmd_[13] = (angle      ) % 10 + '0';


    ROS_INFO_STREAM("in: "<<linear<<", is "<<robot_speed_cmd_[ 4]
                                           <<robot_speed_cmd_[ 5]
                                           <<robot_speed_cmd_[ 6]
                                           <<robot_speed_cmd_[ 7]<<", linear_speed_x = "<<linear_speed_x);
    ROS_INFO_STREAM("in: "<<angle<<", is "<< robot_speed_cmd_[10]
                                           <<robot_speed_cmd_[11]
                                           <<robot_speed_cmd_[12]
                                           <<robot_speed_cmd_[13]<<", angle_speed_yaw = "<<angle_speed_yaw);
    
    boost::asio::write(*p_sp, boost::asio::buffer(robot_speed_cmd_));



    

    

    
    return true;
}

void RobotBringUp::run(void)
{
    bool res=true;

    ros::Rate loop_rate(LOOP_RATE_HZ);
	while (ros::ok()) 
    {
        ros::spinOnce();
        res=spinOnce();
		// linear_speed_x_ =0.0f;
        // angle_speed_yaw_=0.0f;
		loop_rate.sleep();        
	}

    if(!res)
    {
        ROS_FATAL_STREAM("Error.");
    }

    
}

void RobotBringUp::initDataBuf(void)
{
    

    robot_speed_cmd_[ 0] = '!';
    robot_speed_cmd_[ 1] = 'M';
    robot_speed_cmd_[ 2] = ' ';

    robot_speed_cmd_[ 3] = ' ';
    robot_speed_cmd_[ 4] = '0';
    robot_speed_cmd_[ 5] = '0';
    robot_speed_cmd_[ 6] = '0';
    robot_speed_cmd_[ 7] = '0';

    robot_speed_cmd_[ 8] = ' ';

    robot_speed_cmd_[ 9] = ' ';
    robot_speed_cmd_[10] = '0';
    robot_speed_cmd_[11] = '0';
    robot_speed_cmd_[12] = '0';
    robot_speed_cmd_[13] = '0';

    robot_speed_cmd_[14] = 0x0d;

}

}

#endif      //__NEU_ROBOT__