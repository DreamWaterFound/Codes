//测试功能：尝试自己手动“模拟”关键帧和普通帧被创建的过程，并且在视野中进行显示
//不过目前准备首先实现的功能的就是，能够通过控制面板中的参数变化来控制窗口中帧的位姿变化
//可能首先需要解决从Eigen欧拉角=>旋转矩阵，这样的一个变换

//NOTICE 但是这个有问题，恐怕还是得使用矩阵的那种变换形式比较好。
#include <pangolin/pangolin.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

#define PI (3.1415926535897932346f)


using namespace std;

//COL MAJOR!!!
/*
vector<GLfloat> Twc=
{
    1,0,0,0,   
    0,1,0,0,
    0,0,1,0,
    5,0,0,1         //Trans
};
*/

typedef struct __Pose
{

    __Pose(double _pitch,double _roll,double _yaw,double _x,double _y,double _z,bool key=false):
        pitch(_pitch),roll(_roll),yaw(_yaw),x(_x),y(_y),z(_z),keyFrame(key)
        {}

    double pitch;
    double roll;
    double yaw;
    
    double x;
    double y;
    double z;

    bool keyFrame;
}Pose;

void drawFrame(const float w=2.0f);
double deg2rad(const double deg);
Eigen::Matrix3d degEuler2matrix(double pitch,double roll,double yaw);
vector<GLfloat> eigen2glfloat(Eigen::Isometry3d T);
void drawAllFrames(vector<Pose> frames,bool orderRT=true);


int main( int /*argc*/, char** /*argv*/ )
{

    vector<Pose> frames;


    //========================= 窗口 ========================
    pangolin::CreateWindowAndBind(
        "Frame - simulation",     //窗口标题
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
            -20,-20,-20,            //相机光心位置,NOTICE z轴不要设置为0
            0,0,0,              //相机要看的位置
            pangolin::AxisY)    //和观察的方向有关
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

    pangolin::Var<bool> checkOrderBtn("ui.Order_RT",true,true);  
    pangolin::Var<double> frameRoll("ui.frame_roll",0,-90,90);
    pangolin::Var<double> framePitch("ui.frame_pitch",0,-90,90);
    pangolin::Var<double> frameYaw("ui.frame_yaw",0,-180,180);
    pangolin::Var<double> frameX("ui.frame_X",0,-100,100);
    pangolin::Var<double> frameY("ui.frame_Y",0,-100,100);
    pangolin::Var<double> frameZ("ui.frame_Z",0,-100,100);

    
    pangolin::Var<bool> addFrameBtn("ui.add_frame",false,false);
    pangolin::Var<bool> addKeyFrameBtn("ui.add_keyFrame",false,false);
    

    pangolin::Var<bool> resetFrameBtn("ui.reset_frame",false,false);  
    pangolin::Var<bool> resetViewBtn("ui.reset_view",false,false);  
    

    while( !pangolin::ShouldQuit() )
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        //按钮的响应 - 复位视图
        if(pangolin::Pushed(resetViewBtn))
        {
            s_cam.SetModelViewMatrix(
                pangolin::ModelViewLookAt(
            -50,-50,-50,  0,0,0,  pangolin::AxisNegY));

            cout<<"Reset view."<<endl;
        }

        //按钮的响应 - 复位帧的位姿
        if(pangolin::Pushed(resetViewBtn))
        {
            //BUG 目前的问题是，就算是数值复位了，但是控制面板上的gui并不会复位
            framePitch=0.0f;
            framePitch.GuiChanged();
            frameRoll=0.0f;
            frameYaw=0.0f;
            frameX=0.0f;
            frameY=0.0f;
            frameZ=0.0f;

            cout<<"Reset frame pose."<<endl;          
        }

        //按钮的相应- 添加帧
        if(pangolin::Pushed(addFrameBtn))
        {
            frames.push_back(Pose(
                (double)framePitch,
                (double)frameRoll,
                (double)frameYaw,
                (double)frameX,
                (double)frameY,
                (double)frameZ
            ));
        }

        if(pangolin::Pushed(addKeyFrameBtn))
        {
            frames.push_back(Pose(
                (double)framePitch,
                (double)frameRoll,
                (double)frameYaw,
                (double)frameX,
                (double)frameY,
                (double)frameZ,
                true
            ));
        }


        

      
        
        //尝试按照谢晓佳的视频中给出的代码绘制
        pangolin::glDrawAxis((double)axisSize);

        if(checkOrderBtn)
        {
            glRotatef((double)framePitch,1.0,0.0,0.0);
            glRotatef((double)frameRoll,0.0,0.0,1.0);   
            glRotatef((double)frameYaw,0.0,1.0,0.0);
            glTranslatef((double)frameX,(double)frameY,(double)frameZ);
        }
        else
        {
            glTranslatef((double)frameX,(double)frameY,(double)frameZ);
            glRotatef((double)framePitch,1.0,0.0,0.0);
            glRotatef((double)frameRoll,0.0,0.0,1.0);   
            glRotatef((double)frameYaw,0.0,1.0,0.0);
        }


        //绘制帧
        glPushMatrix();
        //glMultMatrixf(Twc.data());
        glColor3f(1.0f,1.0f,1.0f);
        drawFrame();
        drawAllFrames(frames,(bool)checkOrderBtn);
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

double deg2rad(const double deg)
{
    return deg/180.0f*PI;
}


Eigen::Matrix3d degEuler2matrix(double pitch,double roll,double yaw)
{
    Eigen::Vector3d rotation_vector(
            deg2rad((double)yaw),
            deg2rad((double)pitch),
            deg2rad((double)roll));
    Eigen::Matrix3d rotation_matrix=Eigen::Matrix3d::Identity();
    rotation_matrix=Eigen::AngleAxisd(rotation_vector[0],Eigen::Vector3d::UnitZ())
        *Eigen::AngleAxisd(rotation_vector[1],Eigen::Vector3d::UnitY())
        *Eigen::AngleAxisd(rotation_vector[2],Eigen::Vector3d::UnitX());

    return rotation_matrix;

}

vector<GLfloat> eigen2glfloat(Eigen::Isometry3d T)
{
    //注意是列优先
    vector<GLfloat> res;
    for(int j=0;j<4;j++)
    {
        for(int i=0;i<4;i++)
        {
            res.push_back(T(i,j));
        }
    }
    return res;
}


void drawAllFrames(vector<Pose> frames,bool orderRT)
{
    size_t n=frames.size();

    for(int i=0;i<n;i++)
    {
        if(frames[i].keyFrame)
        {
            glColor3f(1.0f,0.0f,0.0f);
        }
        else
        {
            glColor3f(0.0f,1.0f,0.0f);
        }
        
        if(orderRT)
        {
            glRotatef(frames[i].pitch,1.0,0.0,0.0);
            glRotatef(frames[i].roll,0.0,0.0,1.0);   
            glRotatef(frames[i].yaw,0.0,1.0,0.0);
            glTranslatef(frames[i].x,frames[i].y,frames[i].z);

            drawFrame();
        }
        else
        {
            glTranslatef(frames[i].x,frames[i].y,frames[i].z);
            glRotatef(frames[i].pitch,1.0,0.0,0.0);
            glRotatef(frames[i].roll,0.0,0.0,1.0);   
            glRotatef(frames[i].yaw,0.0,1.0,0.0);
            drawFrame();
        }
    }
}