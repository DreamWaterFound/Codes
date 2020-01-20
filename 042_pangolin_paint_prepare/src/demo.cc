/**
 * @file demo.cc
 * @author guoqing (1337841346@qq.com)
 * @brief Pangolin 中 使用OpenGL绘图的例程
 * @version 0.1
 * @date 2019-08-08
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include <iostream>
#include <pangolin/pangolin.h>

#include "demo.hpp"

using namespace std;


int main(int argc, char* argv[])
{
    cout<<"This is a demo for drawing 3D Objects using OpenGL with Pangolin."<<endl;
    cout<<"Complied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    pangolin::CreateWindowAndBind(                                  // 创建窗口
        "demo",                                                     // 窗口标题
        640,                                                        // 窗口高度
        480);                                                       // 窗口宽度
    glEnable(GL_DEPTH_TEST);                                        // 默认开启

    
    pangolin::OpenGlRenderState s_cam(                              // 设置渲染器状态
        pangolin::ProjectionMatrix(                                 // 设置渲染时虚拟相机参数
            640,480,                                                // 相机图像的长和宽
            420,420,320,240,                                        // 相机的内参,fu fv u0 v0
            0.2,100),                                               // 相机所能够看到的最浅和最深的像素
        pangolin::ModelViewLookAt(                                  // 设置相机初始观察的位置
            -2, 2,-2,                                               // 相机初始光心位置,NOTICE z轴不要设置为0
            0,  0, 0,                                               // 相机初始要看的位置
            pangolin::AxisY)                                        // 和观察的方向有关, 现在是设置为左手系
    );

    pangolin::Handler3D handler(s_cam);                             // 创建3D交互对象,实现鼠标对 view 的控制
    pangolin::View& d_cam = pangolin::CreateDisplay()               // 创建 view , 类比于相机的成像平面,是个二维视图
            .SetBounds(                                             // 设置 view 中查看区域的大小
                0.0, 1.0, 0.0, 1.0,                                 // 表示整个窗口都可以观测到,小数表示比例,范围0~1
                -640.0f/480.0f)                                     // 窗口的比例
            .SetHandler(&handler);                                  // 绑定刚刚创建的3D交互对象和 view

    // 主循环中不断刷新显示的内容
    while( !pangolin::ShouldQuit() )
    {
        // 准备清空视图
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // OpenGL 中的着色顺序: RGBA, 每个通道数值 0~1, 数值越大表示该通道颜色越深(对于最后的透明度则表示完全不透明)
        glClearColor(0.2f,0.2f,0.2f,1.0f);

        // ============================== 正式的绘图区域 ================================
        
        // 这些是一些 Pangolin 自带的快速绘图函数, 可以绘制一些基本图形
        
        // 绘制一个 cube
        // pangolin::glDrawColouredCube();

        // 绘制前可以指定颜色\线宽
        glColor3f(0.3f,0.3f,0.3f);
        glLineWidth(0.1);
        // 绘制网格平面, 有利于调试绘图效果的时候找到空间感
        pangolin::glDraw_y0(1.0, 100);                              // xOy平面; 第一个参数表示绘制平面的每个网格的边长, 第二个参数表示网格数量
        // pangolin::glDraw_x0(1.0, 100);
        // pangolin::glDraw_z0(1.0, 100);

        

        glLineWidth(2);
        // 绘制坐标轴, 红 = x轴 绿 = y轴 蓝 = z轴. 参数表示线长
        pangolin::glDrawAxis(1.0);

        // 点绘制 demo
        demoDrawPoint();

        // 绘制多边形 -- 不过这个还用不到
        demoPolygon();

        Eigen::Isometry3d Twc=Eigen::Isometry3d::Identity();
        // 平移变换
        Twc(0,3)=3;
        Twc(1,3)=3;
        Twc(2,3)=3;

        // 绘制八面体线框模型
        drawOctahedronFrame(0.3,0.1,Twc);

        Twc(0,3)=-3;
        Twc(1,3)=-3;
        Twc(2,3)=-3;
        // 绘制八面体线框模型， 另外一种更快的方式， 也是推荐的方式
        drawOctahedronFrame2(0.3,0.1,Twc);

        Twc(0,3)=-3;
        Twc(1,3)=-3;
        Twc(2,3)=3;
        // 演示，绘制八面体的三角片模型，由于没有添加光照所以显示起来很“假”
        // 这个功能不要求做，只做线框的模型
        drawOctahedronTriangles(0.3,0.1,Twc);

        // 结束绘制
        glFlush();
        pangolin::FinishFrame();
    }
    


    return 0;
}