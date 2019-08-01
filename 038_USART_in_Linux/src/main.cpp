#include <iostream>
#include <vector>

#include <boost/asio.hpp>


using namespace std;


boost::asio::io_service iosev;
boost::asio::serial_port sp(iosev, "/dev/ttyUSB0");


int main(int argc, char* argv[])
{
    cout<<"A usart test."<<endl;
    cout<<"Compleied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    cout<<"Opening ttyUSB0 ..."<<endl;

    sp.set_option(boost::asio::serial_port::baud_rate(115200));
	sp.set_option(boost::asio::serial_port::flow_control(boost::asio::serial_port::flow_control::none));
	sp.set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none));
	sp.set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one));
	sp.set_option(boost::asio::serial_port::character_size(8));

    cout<<"OK! "<<endl<<endl;


    // 接收   
    unsigned char buf[8]={0};
   	boost::asio::read(sp, boost::asio::buffer(buf));

    cout<<buf<<endl;

    cout<<endl<<"Terminated."<<endl;


    // 发送
    // char c;
    // unsigned char buf[1]={0};

    // while(1)
    // {
    //     cin>>c;
    //     buf[1]=c;
    //     boost::asio::write(sp, boost::asio::buffer(buf));
    // }

    return 0;
}