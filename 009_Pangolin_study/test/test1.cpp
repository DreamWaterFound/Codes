//测试功能:在窗口中绘制一个坐标轴和点,
//然后可以通过panel中的变量来控制坐标轴的大小和点的大小,以及点的颜色

#include <pangolin/pangolin.h>

int main( int /*argc*/, char** /*argv*/ )
{
    //========================= 窗口 ========================
    pangolin::CreateWindowAndBind(
        "Axis and points test",     //窗口标题
        640,        //窗口尺寸
        480);       //窗口尺寸
    glEnable(GL_DEPTH_TEST);

    //========================== 3D 交互器 =====================
    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(
            640,480,            //相机图像的长和宽
            420,420,320,240,    //相机的内参,fu fv u0 v0
            0.2,100),           //相机所能够看到的最浅和最深的像素
        pangolin::ModelViewLookAt(
            -2,2,-2,            //相机光心位置,NOTICE z轴不要设置为0
            0,0,0,              //相机要看的位置
            pangolin::AxisNegY)    //和观察的方向有关
    );

    //======================== 调节面板 ==============================
     const int UI_WIDTH=180;
     pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::CreatePanel("ui")
      .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    //接下来要开始准备添加控制选项了
    //pangolin::Var<double> axisSize("ui.axis_size",5,0.5,20);
    pangolin::Var<double> pointSize("ui.point_size",3,1,20);
    pangolin::Var<double> pointR("ui.point_r",1,0,1);
    pangolin::Var<double> pointG("ui.point_g",1,0,1);
    pangolin::Var<double> pointB("ui.point_b",1,0,1);
    

    while( !pangolin::ShouldQuit() )
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // Render OpenGL Cube
        //pangolin::glDrawColouredCube();

        //尝试按照谢晓佳的视频中给出的代码绘制
        pangolin::glDrawAxis(3);

        glPointSize(double(pointSize));
        glBegin(GL_POINTS);
        glColor3f(
            (double)pointR,
            (double)pointG,
            (double)pointB);
        glVertex3f(0.0f,0.0f,0.0f);
        glVertex3f(1,0,0);
        glVertex3f(0,2,0);
        glEnd();
        //不要忘记了这个东西!!!
        glFlush();


        // Swap frames and Process Events
        pangolin::FinishFrame();
    }
    
    return 0;
}
