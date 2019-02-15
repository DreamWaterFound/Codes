#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;



//函数原型
cv::Mat generateGaussianTemplate(const Size ksize, const double sigma);
bool selfGaussianBlur(const Mat src, Mat& dst,const Size ksize,const double sigma,BorderTypes bt=BorderTypes::BORDER_REFLECT);


int main(int argc,char *argv[])
{

    if(argc!=2)
    {
        cout<<"Usage:"<<endl;
        cout<<argv[0]<<" img_path"<<endl;
        return 0;
    }

    Mat img=imread(argv[1]);
    if(img.empty())
    {
        cout<<"Img "<<argv[1]<<" is empty!"<<endl;
        return 0;
    }
    Mat grayImg;
    cvtColor(img,grayImg,CV_BGR2GRAY);

    Mat img2;
    GaussianBlur(grayImg,img2,Size(5,5),0.8,0.8);
    imshow("Origin Image",grayImg);
    imshow("OpenCV Guassian Blur",img2);

    Mat img3;
    selfGaussianBlur(grayImg,img3,Size(5,5),0.8);
    imshow("Self Guassian Blur",img3);

    waitKey(0);

    destroyAllWindows();


    return 0;
}



cv::Mat generateGaussianTemplate(const Size ksize, const double sigma)
{
    //create template
    cv::Mat tmp(ksize,CV_64FC1);

    //遍历模板大小,生成高斯函数的值
    double x2,y2;
    double mean_x=ksize.width/2.0;
    double mean_y=ksize.height/2.0;
    double sum=0;
    for(int x=0;x<ksize.width;x++)
    {
        x2=(x-mean_x)*(x-mean_x);
        for(int y=0;y<ksize.height;y++)
        {
            y2=(y-mean_y)*(y-mean_y);
            
            //计算高斯函数数值
            //NOTICE 注意这里使用了一个小技巧,由于后面有归一化过程,会消除这里所乘上的任何常数,所以这里并没有计算恼人的PI
            tmp.at<double>(y,x)=exp(-(x2+y2)/(sigma*sigma));
            sum+=tmp.at<double>(y,x);
        }
    }


    //然后记得归一化,这里选择是是按中心进行归一化
    //计算归一化系数
    double s=1.0/sum;
    for(int x=0;x<ksize.width;x++)
    {
        for(int y=0;y<ksize.height;y++)
        {
            tmp.at<double>(y,x)*=s;
        }
    }

    return tmp;
}

bool selfGaussianBlur(const Mat src, Mat& dst,const Size ksize,const double sigma,BorderTypes bt)
{
    int ch=src.channels();
    dst=src.clone();
    if(ch!=1&&ch!=3)
    {
        return false;
    }

    //首先生成高斯滤波器的模板
    Mat tmp=generateGaussianTemplate(ksize,sigma);


    //生成图像边界 
    //NOTE 其实从这里就可以看出来, 图像处理中所说的"图像边界"并不是我们想当然所以为的"图像边界"
    Size border_width(ksize.width/2,ksize.height/2);
    Mat bimg;
    copyMakeBorder(src,bimg,border_width.height,border_width.height,border_width.width,border_width.width,bt);

    imshow("Extened Border",bimg);
    waitKey(0);
    //进行卷积操作前,先需要获取一些图像的参数
    //TODO 目前先只处理灰度图像的情况
    
    int max_x=bimg.cols-border_width.width;
    int max_y=bimg.rows-border_width.height;
    //开始遍历图像上的每一个像素
    for(int x=border_width.width;x<max_x;x++)
    {
        for(int y=border_width.height;y<max_y;y++)
        {
            //累加卷积值
            double sum[3]={0};

            //对于单色图像的情况
            if(ch==1)
            {
                for(int m=-border_width.width;m<border_width.width;m++)
                {   
                    for(int n=-border_width.height;n<border_width.height;n++)
                    {
                        sum[0] += tmp.at<double>(border_width.height+n,border_width.width+m) * 
                                  bimg.at<unsigned char>(y+n,x+m);
                    }
                }    

                if(sum[0]<0)    sum[0]=0;
                if(sum[0]>255) sum[0]=255;

                dst.at<unsigned char>(y-border_width.height,x-border_width.width)=sum[0];
            }                   
        }
    }

    return true;

}

/*
REVIEW 感觉参考的程序中有几个问题不太对:
1. 在进行卷积的过程中,在某个像素的遍历后会覆写这个像素;这样子不会导致后面依赖于这个像素的卷积过程出现错误吗?因此感觉这个应该是有问题的;
2. dst这个图像中是保存了扩充边界的结果的;如果直接返回这个dst结果的话,滤波器输出的图像边缘也会有这个扩充的边界.这个是不是应该进行去除?
*/
