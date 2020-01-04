// STL
#include <iostream>
#include <string>
// System
#include <stdlib.h>
// OpenCV
#include <opencv2/opencv.hpp>
// OpenNI
#include <OpenNI.h>

// 命名空间
using namespace cv;
using namespace std;
using namespace openni;

/**
 * @brief 检查OpenNI的错误
 * @param[in] result 错误码
 * @param[in] status 当前所处阶段
 */
void CheckOpenNIError( Status result, string status )
{
    if( result != STATUS_OK )
        // 输出详细的错误提示信息
        cerr << status << " Error: " << OpenNI::getExtendedError() << endl;
}

// 主函数
int main( int argc, char** argv )
{
    // step 0 数据准备
    cout<<"OpenNI 2 test."<<endl;
    cout<<"Complied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    openni::Version version = openni::OpenNI::getVersion();
		std::cout << version.minor << "."
				<< version.major << "."
				<< version.maintenance << "."
				<< version.build 
				<< std::endl;


    Status result = STATUS_OK;

    //OpenNI2 image
    VideoFrameRef oniDepthImg;
    VideoFrameRef oniColorImg;

    //OpenCV image
    cv::Mat cvDepthImg;
    cv::Mat cvBGRImg;
    cv::Mat cvFusionImg;

    cv::namedWindow("depth");
    cv::namedWindow("image");
    cv::namedWindow("fusion");
    char key=0;

    // step 1 initialize OpenNI2 and open device
    result = OpenNI::initialize();
    CheckOpenNIError( result, "initialize context" );
    // open device
    Device device;
    result = device.open( openni::ANY_DEVICE );
    CheckOpenNIError( result, "open device" );

    // step 2 create, config and start depth stream
    VideoStream oniDepthStream;
    result = oniDepthStream.create( device, openni::SENSOR_DEPTH );
    // set depth video mode
    VideoMode modeDepth;
    modeDepth.setResolution( 640, 480 );
    modeDepth.setFps( 30 );
    modeDepth.setPixelFormat( PIXEL_FORMAT_DEPTH_1_MM );
    oniDepthStream.setVideoMode(modeDepth);
    // start depth stream
    result = oniDepthStream.start();

    // step 3 create, config and start color stream
    VideoStream oniColorStream;
    result = oniColorStream.create( device, openni::SENSOR_COLOR );
    // set color video mode
    VideoMode modeColor;
    modeColor.setResolution( 640, 480 );
    modeColor.setFps( 30 );
    modeColor.setPixelFormat( PIXEL_FORMAT_RGB888 );
    oniColorStream.setVideoMode( modeColor);
    // set depth and color imge registration mode
    // 看来有的相机是不支持反向的配准的
    if( device.isImageRegistrationModeSupported(IMAGE_REGISTRATION_DEPTH_TO_COLOR ) )
    {
        device.setImageRegistrationMode( IMAGE_REGISTRATION_DEPTH_TO_COLOR );
    }
    // start color stream
    result = oniColorStream.start();

    // step 4 main loop
    while( key!=27 )
    {
        // step 4.1 read color and depth frame
        if( oniColorStream.readFrame( &oniColor Img ) == STATUS_OK )
        {
            // convert data into OpenCV type
            cv::Mat cvRGBImg( oniColorImg.getHeight(), oniColorImg.getWidth(), CV_8UC3, (void*)oniColorImg.getData() );
            cv::cvtColor( cvRGBImg, cvBGRImg, CV_RGB2BGR );
            cv::imshow( "image", cvBGRImg );
        }

        if( oniDepthStream.readFrame( &oniDepthImg ) == STATUS_OK )
        {
            cv::Mat cvRawImg16U( oniDepthImg.getHeight(), oniDepthImg.getWidth(), CV_16UC1, (void*)oniDepthImg.getData() );
            // convert depth image GRAY to BGR
            cvRawImg16U.convertTo( cvDepthImg, CV_8U, 255.0/(oniDepthStream.getMaxPixelValue()));
            cv::cvtColor(cvDepthImg,cvFusionImg,CV_GRAY2BGR);
            cv::imshow( "depth", cvDepthImg );
        }
        // step 4.2 生成 Fusion 后的图像，并显示
        cv::addWeighted(cvBGRImg,0.5,cvFusionImg,0.5,0,cvFusionImg);
        cv::imshow( "fusion", cvFusionImg );
        key = cv::waitKey(20);
    }

    // step 5 善后
    //cv destroy
    cv::destroyWindow("depth");
    cv::destroyWindow("image");
    cv::destroyWindow("fusion");

    //OpenNI2 destroy
    oniDepthStream.destroy();
    oniColorStream.destroy();
    device.close();
    OpenNI::shutdown();

    return 0;
}