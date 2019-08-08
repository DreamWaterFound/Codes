/**
 * @file demo.hpp
 * @author guoqing (1337841346@qq.com)
 * @brief 实例demo.h
 * @version 0.1
 * @date 2019-08-08
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#ifndef __DEMO_HPP__
#define __DEMO_HPP__

#include <vector>
#include <pangolin/pangolin.h>
#include <Eigen/Core>


// 一个转换函数
std::vector<GLfloat> eigen2glfloat(Eigen::Isometry3d T)
{
    //注意是列优先
    std::vector<GLfloat> res;
    for(int j=0;j<4;j++)
    {
        for(int i=0;i<4;i++)
        {
            res.push_back(T(i,j));
        }
    }
    return res;
}



void demoDrawPoint(void)
{
    // 设置点大小和颜色
    glPointSize(5.0f);
    glColor3f(1.0,1.0,1.0);
    // 每次绘制前都要有这个; GL_POINTS 表示当前绘制点
    // ref: https://blog.csdn.net/aa941096979/article/details/50843596
    glBegin(GL_POINTS);
    // 原点
    glVertex3f(0.0f,0.0f,0.0f);
    // x轴上的一个点
    glVertex3f(0.5,0,0);
    // y轴上的一个点
    glVertex3f(0,1,0);
    glEnd();
}


void demoPolygon(void)
{
    glColor3f(0.0, 0.54,0.54);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    glBegin(GL_POLYGON);
    glVertex3f(2.25,0.25,0.4);
    glVertex3f(2.75,2.25,0.4);
    glVertex3f(2.75,0.75,-0.9);
    glVertex3f(5.0,0.75,0.4);
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
}

void drawOctahedronFrame(double size, double line_width, Eigen::Isometry3d& Twc)
{
    // 先绘制图像,再转移变换到任意位置的示例
    std::vector<GLfloat> glTwc=eigen2glfloat(Twc);


    // 绘制参数, 正方形边长 2a, 高度b
    const double a=size;
    const double b=a*sqrt(2);

    // 要和后面的 popMatrix 成对出现
    glPushMatrix();
    glMultMatrixf((GLfloat*)glTwc.data());

    glLineWidth(line_width);
    // 颜色
    glColor3f(0.8f,0.54f,0.78f);
    // 绘制直线,每两个点组成一个线段
    glBegin(GL_LINES);

    // 绘制正方形
    glVertex3f(a,a,0);
    glVertex3f(a,-a,0);
    glVertex3f(a,-a,0);
    glVertex3f(-a,-a,0);
    glVertex3f(-a,-a,0);
    glVertex3f(-a,a,0);
    glVertex3f(-a,a,0);
    glVertex3f(a,a,0);


    // 连接上部顶点
    glVertex3f(a,a,0);
    glVertex3f(0,0,b);
    glVertex3f(a,-a,0);
    glVertex3f(0,0,b);
    glVertex3f(-a,-a,0);
    glVertex3f(0,0,b);
    glVertex3f(-a,a,0);
    glVertex3f(0,0,b);

    // 连接下部顶点
    glVertex3f(a,a,0);
    glVertex3f(0,0,-b);
    glVertex3f(a,-a,0);
    glVertex3f(0,0,-b);
    glVertex3f(-a,-a,0);
    glVertex3f(0,0,-b);
    glVertex3f(-a,a,0);
    glVertex3f(0,0,-b);

    glEnd();
    glPopMatrix();
}

// NOTICE
// 这个是最后我们想要的效果;通过调用该函数,指定物体大小、线宽、位姿以及颜色等，就可以绘制出相应物体的线框模型
void drawOctahedronFrame2(double size, double line_width, Eigen::Isometry3d& Twc)
{
    std::vector<GLfloat> glTwc=eigen2glfloat(Twc);

    const double a=size;
    const double b=a*sqrt(2);

    glPushMatrix();
    glMultMatrixf((GLfloat*)glTwc.data());

    glLineWidth(line_width);
    // 颜色
    glColor3f(0.5f,0.8f,0.12f);
    // 绘制直线,所有的点相连组成一组线段,对于圆圈形状,推荐使用这种绘图方式,因为它比较快
    // 还有一个 GL_LINE_STRIP ,不过这个不是首尾相连
    glBegin(GL_LINE_LOOP);
    // 先绘制正方形
    glVertex3f(a,a,0);
    glVertex3f(a,-a,0);
    glVertex3f(-a,-a,0);
    glVertex3f(-a,a,0);
    glEnd();


    glBegin(GL_LINE_LOOP);
    // 继续绘制
    glVertex3f(0,0,b);
    glVertex3f(a,a,0);
    glVertex3f(0,0,-b);
    glVertex3f(-a,-a,0);
    glEnd();

    glBegin(GL_LINE_LOOP);
    // 继续绘制
    glVertex3f(0,0,b);
    glVertex3f(a,-a,0);
    glVertex3f(0,0,-b);
    glVertex3f(-a,a,0);
    glEnd();

    
    glPopMatrix();
}


void drawOctahedronTriangles(double size, double line_width, Eigen::Isometry3d& Twc)
{
    std::vector<GLfloat> glTwc=eigen2glfloat(Twc);

    const double a=size;
    const double b=a*sqrt(2);

    glPushMatrix();
    glMultMatrixf((GLfloat*)glTwc.data());

    glLineWidth(line_width);
    // 颜色
    glColor3f(0.1f,1.0f,0.32f);
    // 绘制三角面片
    glBegin(GL_TRIANGLES);
    // 先绘制正方形上方的四个三角片
    glVertex3f(a,a,0);
    glVertex3f(a,-a,0);
    glVertex3f(0,0,b);

    glVertex3f(a,a,0);
    glVertex3f(-a,a,0);
    glVertex3f(0,0,b);

    glVertex3f(-a,-a,0);
    glVertex3f(a,-a,0);
    glVertex3f(0,0,b);

    glVertex3f(-a,-a,0);
    glVertex3f(-a,a,0);
    glVertex3f(0,0,b);

    // 然后是正方形下方的四个三角片
    glVertex3f(a,a,0);
    glVertex3f(a,-a,0);
    glVertex3f(0,0,-b);

    glVertex3f(a,a,0);
    glVertex3f(-a,a,0);
    glVertex3f(0,0,-b);

    glVertex3f(-a,-a,0);
    glVertex3f(a,-a,0);
    glVertex3f(0,0,-b);

    glVertex3f(-a,-a,0);
    glVertex3f(-a,a,0);
    glVertex3f(0,0,-b);

    glEnd();

    // 为了可视化效果,再绘制线框(和drawOctahedronFrame2中操作相同)
    glColor3f(0.0f,0.0f,0.0f);
    glLineWidth(10*line_width);

    glBegin(GL_LINE_LOOP);
    // 先绘制正方形
    glVertex3f(a,a,0);
    glVertex3f(a,-a,0);
    glVertex3f(-a,-a,0);
    glVertex3f(-a,a,0);
    glEnd();

    glBegin(GL_LINE_LOOP);
    // 继续绘制
    glVertex3f(0,0,b);
    glVertex3f(a,a,0);
    glVertex3f(0,0,-b);
    glVertex3f(-a,-a,0);
    glEnd();

    glBegin(GL_LINE_LOOP);
    // 继续绘制
    glVertex3f(0,0,b);
    glVertex3f(a,-a,0);
    glVertex3f(0,0,-b);
    glVertex3f(-a,a,0);
    glEnd();

    glPopMatrix();

}



#endif // __DEMO_HPP__