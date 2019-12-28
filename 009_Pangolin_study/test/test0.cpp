/**
 * @file test0.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 一个平面实时绘图的demo
 * @version 0.1
 * @date 2019-12-26
 * 
 * @copyright Copyright (c) 2019
 * 
 */


#include <iostream>

#include <pangolin/pangolin.h>

int main(/*int argc, char* argv[]*/)
{
    // step 0 创建显示窗口
    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind("Plot test",1024,768);

    // step 1 创建消息记录器
    // Data logger object
    pangolin::DataLog log;

    // step 2 生成并设置数据标签
    // Optionally add named labels
    std::vector<std::string> labels;
    labels.push_back(std::string("sin(t)"));
    labels.push_back(std::string("cos(t)"));
    labels.push_back(std::string("sin(t)+cos(t)"));
    log.SetLabels(labels);

    // 每次运行的步进长度
    const float tinc = 0.01f;

    // OpenGL 'view' of data. We might have many views of the same data.
    pangolin::Plotter plotter(
        &log,                           // 设置消息记录器
        0.0f,                           // 坐标轴 Left
        4.0f*(float)M_PI/tinc,          // 坐标轴 Right
        -2.0f,                          // 坐标轴 Bottom
        2.0f,                           // 坐标轴 Top
        2.5*(float)M_PI/(4.0f*tinc),    // 和坐标轴的横轴间隔有关系
        0.4f);                          // 和坐标轴的纵轴间隔有关系

    // 设置 plotter 在整个窗口中占据的区域范围
    plotter.SetBounds(0.0, 1.0, 0.0, 1.0);
    // ?
    plotter.Track("$i");

    // HACK

    plotter.SetTickColour(pangolin::Colour(1,1,0,1));

    // // Add some sample annotations to the plot
    // plotter.AddMarker(pangolin::Marker::Vertical,   -1000, pangolin::Marker::LessThan, pangolin::Colour::Blue().WithAlpha(0.2f) );
    // plotter.AddMarker(pangolin::Marker::Horizontal,   100, pangolin::Marker::GreaterThan, pangolin::Colour::Red().WithAlpha(0.2f) );
    // plotter.AddMarker(pangolin::Marker::Horizontal,    10, pangolin::Marker::Equal, pangolin::Colour::Green().WithAlpha(0.2f) );

    plotter.AddMarker(pangolin::Marker::Vertical,   (float)M_PI/tinc, pangolin::Marker::LessThan, pangolin::Colour::Blue().WithAlpha(0.4f) );
    plotter.AddMarker(pangolin::Marker::Horizontal, 2*(float)M_PI/tinc, pangolin::Marker::GreaterThan, pangolin::Colour::Red().WithAlpha(0.4f) );
    plotter.AddMarker(pangolin::Marker::Horizontal, 1.5, pangolin::Marker::Equal, pangolin::Colour::Green().WithAlpha(0.7f) );

    // 生成了 plotter 对象还没有完事, 还得继续让 display 对象对其加以显示
    pangolin::DisplayBase().AddDisplay(plotter);

    // 后面要用到的累加量
    float t = 0;

    // Default hooks for exiting (Esc) and fullscreen (tab).
    while( !pangolin::ShouldQuit() )
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 这里的 log 是记录的意思; 这个函数的参数应该是不定的 -- 实际上是重载了各种可能的参数数据数量
        log.Log(sin(t),cos(t),sin(t)+cos(t));

        // 数据累加
        t += tinc;

        // Render graph, Swap frames and Process Events
        pangolin::FinishFrame();
    }

    return 0;
}
