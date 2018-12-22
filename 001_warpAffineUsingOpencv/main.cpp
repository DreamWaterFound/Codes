#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

//全局变量
String src_windowName = "原图像";
String warp_windowName = "仿射变换";
String warp_rotate_windowName = "仿射旋转变换";
String rotate_windowName = "图像旋转";

int main(int argc,char *agrv[])
{

    cout<<"程序开始运行。"<<endl;

    Point2f srcTri[3];
    Point2f dstTri[3];

    Mat rot_mat(2, 3, CV_32FC1);
    Mat warp_mat(2, 3, CV_32FC1);
    Mat srcImage, warp_dstImage, warp_rotate_dstImage, rotate_dstImage;

    //加载图像
    cout<<"程序当前目录为："<<agrv[0]<<endl;
    srcImage = imread("./img/1.jpg");
    //cout<<"文件目录为： "<< ".\\img\\1.jpg"<<endl;

    //判断文件是否加载成功
    if(srcImage.empty())
    {
        cout << "图像加载失败!" << endl;
        return -1;
    }
    else
        cout << "图像加载成功!" << endl << endl;

    //创建仿射变换目标图像与原图像尺寸类型相同
    warp_dstImage = Mat::zeros(srcImage.rows, srcImage.cols, srcImage.type());

    //设置三个点来计算仿射变换
    srcTri[0] = Point2f(0, 0);
    srcTri[1] = Point2f(srcImage.cols - 1, 0);
    srcTri[2] = Point2f(0, srcImage.rows - 1);

    dstTri[0] = Point2f(srcImage.cols*0.0, srcImage.rows*0.33);
    dstTri[1] = Point2f(srcImage.cols*0.85, srcImage.rows*0.25);
    dstTri[2] = Point2f(srcImage.cols*0.15, srcImage.rows*0.7);

    //计算仿射变换矩阵
    warp_mat = getAffineTransform(srcTri, dstTri);

    //对加载图形进行仿射变换操作
    warpAffine(srcImage, warp_dstImage, warp_mat, warp_dstImage.size());

    //计算图像中点顺时针旋转50度，缩放因子为0.6的旋转矩阵
    Point center = Point(warp_dstImage.cols/2, warp_dstImage.rows/2);
    double angle = -50.0;
    double scale = 0.6;

    //计算旋转矩阵
    rot_mat = getRotationMatrix2D(center, angle, scale);

    //旋转已扭曲图像
    warpAffine(warp_dstImage, warp_rotate_dstImage, rot_mat, warp_dstImage.size());

    //将原图像旋转
    warpAffine(srcImage, rotate_dstImage, rot_mat, srcImage.size());

    //显示变换结果
    namedWindow(src_windowName, WINDOW_AUTOSIZE);
    imshow(src_windowName, srcImage);

    namedWindow(warp_windowName, WINDOW_AUTOSIZE);
    imshow(warp_windowName, warp_dstImage);

    namedWindow(warp_rotate_windowName, WINDOW_AUTOSIZE);
    imshow(warp_rotate_windowName, warp_rotate_dstImage);

    namedWindow(rotate_windowName, WINDOW_AUTOSIZE);
    imshow(rotate_windowName, rotate_dstImage);

    waitKey(0);

    return 0;
}

//原文：https://blog.csdn.net/keith_bb/article/details/56331356 
