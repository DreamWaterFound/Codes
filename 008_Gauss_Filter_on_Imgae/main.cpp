#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;



//函数原型
cv::Mat generateGaussianTemplate(const Size ksize, const double sigma);
bool selfGaussianBlur(const Mat src, Mat& dst,const Size ksize,const double sigma,BorderTypes bt=BorderTypes::BORDER_REFLECT);

//分离的高斯滤波器
cv::Mat generateGaussianTemplate1D(const size_t size, const double sigma);
bool selfDGaussianBlur(const Mat src, Mat& dst,const Size ksize,const double sigma_x, const double sigma_y,BorderTypes bt=BorderTypes::BORDER_REFLECT);


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
        cout<<"Img "<<argv[1]<<" is0empty!"<<endl;
        return 0;
    }
    Mat grayImg;
    cvtColor(img,grayImg,CV_BGR2GRAY);

    Mat img2;

    auto start = system_clock::now();
    GaussianBlur(grayImg,img2,Size(0,5),0.8,0.8);
    auto end = system_clock::now();
    auto duration_opencv = duration_cast<microseconds>(end-start);
    imshow("Origin Image",img);
    imshow("OpenCV Guassian Blur",img2);

    Mat img3;
    start=system_clock::now();
    selfDGaussianBlur(img,img3,Size(5,5),0.8,0.8);
    end=system_clock::now();
    auto duration_myself2 = duration_cast<microseconds>(end-start);
    imshow("Self Guassian Blur",img3);

    start=system_clock::now();
    selfGaussianBlur(grayImg,img3,Size(5,5),0.8);
    end=system_clock::now();
    auto duration_myself1 = duration_cast<microseconds>(end-start);

    cout<<"Time cost:"<<endl;
    cout<<"\tOpenCV method: \t\t\t\t "<<double(duration_opencv.count())*microseconds::period::num/microseconds::period::den<<" s;"<<endl;
    cout<<"\tMyself method (without separate):\t "<<double(duration_myself1.count())*microseconds::period::num/microseconds::period::den<<" s;"<<endl;
    cout<<"\tMyself method (with separate):\t\t "<<double(duration_myself2.count())*microseconds::period::num/microseconds::period::den<<" s."<<endl;

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

    //imshow("Extened Border",bimg);
    //waitKey(0);
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
            else
            {
                for(int m=-border_width.width;m<border_width.width;m++)
                {   
                    for(int n=-border_width.height;n<border_width.height;n++)
                    {
                        Vec3b rgb = bimg.at<Vec3b>(y+n,x+m);
                        auto t=tmp.at<double>(border_width.height+n,border_width.width+m);
                        sum[0] +=  t*rgb[0];
                        sum[1] +=  t*rgb[1];
                        sum[2] +=  t*rgb[2];
                    }
                }    

                if(sum[0]<0)    sum[0]=0;
                if(sum[0]>255) sum[0]=255;

                if(sum[1]<0)    sum[1]=0;
                if(sum[1]>255) sum[1]=255;

                if(sum[2]<0)    sum[2]=0;
                if(sum[2]>255) sum[2]=255;

                Vec3b rgb={static_cast<uchar>(sum[0]),
                           static_cast<uchar>(sum[1]),
                           static_cast<uchar>(sum[2])};
                dst.at<Vec3b>(y-border_width.height,x-border_width.width)=rgb;
                
                
            }
            
        }
    }

    return true;

}


//========================  分离的高斯滤波器  =================================

bool selfDGaussianBlur(const Mat src, Mat& dst,const Size ksize,const double sigma_x, const double sigma_y,BorderTypes bt)
{
    int ch=src.channels();
    dst=src.clone();
    if(ch!=1&&ch!=3)
    {
        return false;
    }

    //首先生成高斯滤波器的模板
    Mat tmp_x=generateGaussianTemplate1D(ksize.width,sigma_x);
    Mat tmp_y=generateGaussianTemplate1D(ksize.height,sigma_y);
    


    //生成图像边界 
    //NOTE 其实从这里就可以看出来, 图像处理中所说的"图像边界"并不是我们想当然所以为的"图像边界"
    Size border_width(ksize.width/2,ksize.height/2);
    Mat bimg;
    copyMakeBorder(src,bimg,border_width.height,border_width.height,border_width.width,border_width.width,bt);

    //imshow("Extened Border",bimg);
    //waitKey(0);
    //进行卷积操作前,先需要获取一些图像的参数
    //TODO 目前先只处理灰度图像的情况
    
    int max_x=bimg.cols-border_width.width;
    int max_y=bimg.rows-border_width.height;

    //开始遍历像素进行滤波。首先是横向滤波
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
                    sum[0] += tmp_x.at<double>(border_width.width+m) * bimg.at<unsigned char>(y,x+m);
                }    

                if(sum[0]<0)    sum[0]=0;
                if(sum[0]>255) sum[0]=255;

                dst.at<unsigned char>(y-border_width.height,x-border_width.width)=sum[0];
            }                   
            else
            {
                for(int m=-border_width.width;m<border_width.width;m++)
                {   
                    Vec3b rgb = bimg.at<Vec3b>(y,x+m);
                    auto t=tmp_x.at<double>(border_width.width+m);
                    sum[0] +=  t*rgb[0];
                    sum[1] +=  t*rgb[1];
                    sum[2] +=  t*rgb[2];
                }    

                if(sum[0]<0)    sum[0]=0;
                if(sum[0]>255) sum[0]=255;

                if(sum[1]<0)    sum[1]=0;
                if(sum[1]>255) sum[1]=255;

                if(sum[2]<0)    sum[2]=0;
                if(sum[2]>255) sum[2]=255;

                Vec3b rgb={static_cast<uchar>(sum[0]),
                           static_cast<uchar>(sum[1]),
                           static_cast<uchar>(sum[2])};
                dst.at<Vec3b>(y-border_width.height,x-border_width.width)=rgb;
                
                
            }//分颜色通道进行讨论
        }//见下
    }//遍历每一个图像像素

    //第二遍遍历图像,这个时候卷积计算在y轴方向上的高斯核
    for(int x=border_width.width;x<max_x;x++)
    {
        for(int y=border_width.height;y<max_y;y++)
        {
            //累加卷积值
            double sum[3]={0};

            //对于单色图像的情况
            if(ch==1)
            {
                for(int n=-border_width.height;n<border_width.height;n++)
                {               
                    sum[0] += tmp_y.at<double>(border_width.height+n) * bimg.at<unsigned char>(y+n,x);
                }    

                if(sum[0]<0)    sum[0]=0;
                if(sum[0]>255) sum[0]=255;

                dst.at<unsigned char>(y-border_width.height,x-border_width.width)=sum[0];
            }                   
            else
            {
                for(int n=-border_width.height;n<border_width.height;n++)
                {
                    Vec3b rgb = bimg.at<Vec3b>(y+n,x);
                    auto t=tmp_y.at<double>(border_width.height+n);
                    sum[0] +=  t*rgb[0];
                    sum[1] +=  t*rgb[1];
                    sum[2] +=  t*rgb[2];
                }

                if(sum[0]<0)    sum[0]=0;
                if(sum[0]>255) sum[0]=255;

                if(sum[1]<0)    sum[1]=0;
                if(sum[1]>255) sum[1]=255;

                if(sum[2]<0)    sum[2]=0;
                if(sum[2]>255) sum[2]=255;

                Vec3b rgb={static_cast<uchar>(sum[0]),
                           static_cast<uchar>(sum[1]),
                           static_cast<uchar>(sum[2])};
                dst.at<Vec3b>(y-border_width.height,x-border_width.width)=rgb;                
            }//分颜色通道进行讨论
        }//见下
    }//遍历每一个图像像素

    return true;

}

//产生一个一维的高斯模板
cv::Mat generateGaussianTemplate1D(const size_t size, const double sigma)
{
    //create template
    cv::Mat tmp(1,size,CV_64FC1);

    //遍历模板大小,生成高斯函数的值
    double x2;
    double mean_x=size/2.0;
    double sum=0;
    for(int x=0;x<size;x++)
    {
        x2=(x-mean_x)*(x-mean_x);
        tmp.at<double>(x)=exp(-x2/(2*sigma*sigma));
        sum+=tmp.at<double>(x);
    }

    //然后记得归一化,这里选择是是按和进行归一化。计算归一化系数
    double s=1.0/sum;
    for(int x=0;x<size;x++)
    {
        tmp.at<double>(x)*=s;
    }

    return tmp;
}

