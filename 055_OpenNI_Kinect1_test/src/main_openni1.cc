 

#define linux true

#include <stdlib.h>
#include <iostream>
#include <string>
//【1】
#include <XnCppWrapper.h>
#include <XnOpenNI.h>

// #include <>
#include "opencv/cv.h"
#include "opencv/highgui.h"
 
using namespace std;
using namespace cv;
 
void CheckOpenNIError( XnStatus result, string status )
{ 
	if( result != XN_STATUS_OK ) 
		cerr << status << " Error: " << xnGetStatusString( result ) << endl;
}
 
int main( int argc, char** argv )
{
    cout<<"OpenNI 1 test."<<endl;
    cout<<"Complied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;


	XnStatus result = XN_STATUS_OK;  
	xn::DepthMetaData depthMD;
	xn::ImageMetaData imageMD;
 
	//OpenCV
	IplImage*  imgDepth16u=cvCreateImage(cvSize(640,480),IPL_DEPTH_16U,1);
	IplImage* imgRGB8u=cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,3);
	IplImage*  depthShow=cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,1);
	IplImage* imageShow=cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,3);
	cvNamedWindow("depth",1);
	cvNamedWindow("image",1);
	char key=0;
 
	//【2】
	// context 
	xn::Context context; 
	result = context.Init(); 
	CheckOpenNIError( result, "initialize context" );  
 
	// creategenerator  
	xn::DepthGenerator depthGenerator;  
	result = depthGenerator.Create( context ); 
	CheckOpenNIError( result, "Create depth generator" );  
	xn::ImageGenerator imageGenerator;
	result = imageGenerator.Create( context ); 
	CheckOpenNIError( result, "Create image generator" );
 
	//【3】
	//map mode  
	XnMapOutputMode mapMode; 
	mapMode.nXRes = 640;  
	mapMode.nYRes = 480; 
	mapMode.nFPS = 30; 
	result = depthGenerator.SetMapOutputMode( mapMode );  
	result = imageGenerator.SetMapOutputMode( mapMode );  
 
	//【4】
	// correct view port  
	depthGenerator.GetAlternativeViewPointCap().SetViewPoint( imageGenerator ); 
 
	//【5】
	//read data
	result = context.StartGeneratingAll();  
	//【6】
	result = context.WaitNoneUpdateAll();  
 
	while( (key!=27) && !(result = context.WaitNoneUpdateAll( ))  ) 
	{  
		//get meta data
		depthGenerator.GetMetaData(depthMD); 
		imageGenerator.GetMetaData(imageMD);
 
		//【7】
		//OpenCV output
		memcpy(imgDepth16u->imageData,depthMD.Data(),640*480*2);
		cvConvertScale(imgDepth16u,depthShow,255/4096.0,0);
		memcpy(imgRGB8u->imageData,imageMD.Data(),640*480*3);
		cvCvtColor(imgRGB8u,imageShow,CV_RGB2BGR);
		cvShowImage("depth", depthShow);
		cvShowImage("image",imageShow);
		key=cvWaitKey(20);
	}
 
	//destroy
	cvDestroyWindow("depth");
	cvDestroyWindow("image");
	cvReleaseImage(&imgDepth16u);
	cvReleaseImage(&imgRGB8u);
	cvReleaseImage(&depthShow);
	cvReleaseImage(&imageShow);
	context.StopGeneratingAll();
	context.Shutdown();
	return 0;
}