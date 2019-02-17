//测试功能：尝试自己手动“模拟”关键帧和普通帧被创建的过程，并且在视野中进行显示
#include <pangolin/pangolin.h>
#include <vector>
#include <iostream>
using namespace std;

//COL MAJOR!!!
vector<GLfloat> Twc=
{
    1,0,0,0,   
    0,1,0,0,
    0,0,1,0,
    5,0,0,1         //Trans
};


void drawFrame(const float w=2.0f);

int main( int /*argc*/, char** /*argv*/ )
{
    //========================= 窗口 ========================
    pangolin::CreateWindowAndBind(
        "Frame test",     //窗口标题
        640,        //窗口尺寸
        480);       //窗口尺寸
    glEnable(GL_DEPTH_TEST);

    //========================== 3D 交互器 =====================
    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(
            640,480,            //相机图像的长和宽
            420,420,320,240,    //相机的内参,fu fv u0 v0
            0.2,500),           //相机所能够看到的最浅和最深的像素
        pangolin::ModelViewLookAt(
            -50,-50,-50,            //相机光心位置,NOTICE z轴不要设置为0
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
    pangolin::Var<double> axisSize("ui.axis_size",5,1,20);
    pangolin::Var<double> pointSize("ui.point_size",3,1,20);
    pangolin::Var<double> pointR("ui.point_r",1,0,1);
    pangolin::Var<double> pointG("ui.point_g",1,0,1);
    pangolin::Var<double> pointB("ui.point_b",1,0,1);
    pangolin::Var<bool> resetBtn("ui.reset_view",false,false);  
    

    while( !pangolin::ShouldQuit() )
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        if(pangolin::Pushed(resetBtn))
        {
            s_cam.SetModelViewMatrix(
                pangolin::ModelViewLookAt(
            -50,-50,-50,  0,0,0,  pangolin::AxisNegY));

            cout<<"Reset view."<<endl;
        }

        //尝试按照谢晓佳的视频中给出的代码绘制
        pangolin::glDrawAxis((double)axisSize);

        //绘制帧
        glPushMatrix();
        glMultMatrixf(Twc.data());
        drawFrame();
        glPopMatrix();
        
        //不要忘记了这个东西!!!
        glFlush();

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }
    
    return 0;
}

void drawFrame(const float w)
{
    const float h=w*0.75;
    const float z=w*0.6;

    glLineWidth(2);
    glColor3f(1.0f,0.0f,0.0f);

    glBegin(GL_LINES);

    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);
    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);
    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);
    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);

    glEnd();
}
