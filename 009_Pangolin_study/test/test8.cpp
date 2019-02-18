//尝试显示两个图片
#include <pangolin/pangolin.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <thread>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#define PI (3.1415926535897932346f)


using namespace std;
using namespace cv;

static const std::string window_name = "Two images test";

typedef struct __Pose
{
    __Pose(const vector<GLfloat> _Twc,const bool _keyFrame):
        Twc(_Twc),keyFrame(_keyFrame)
        {;}

    bool keyFrame;
    vector<GLfloat> Twc;   
}Pose;

void drawFrame(const float w=2.0f);
double deg2rad(const double deg);
Eigen::Matrix3d degEuler2matrix(double pitch,double roll,double yaw);
vector<GLfloat> eigen2glfloat(Eigen::Isometry3d T);
void drawAllFrames(vector<Pose> frames);

//线程配置相关
void run(void);
void setup(void);

//图像显示相关
void setImageData(unsigned char * imageArray, Mat img){

  for(int i=0;i<img.rows;i++) {
      for(int j=0;j<img.cols;j++)
      {
          
          //Vec3b rgb=img.at<Vec3b>(j,i);
          /*
          imageArray[i*3*img.cols+j]=rgb[0];
          imageArray[i*3*img.cols+j+1]=rgb[1];
          imageArray[i*3*img.cols+j+2]=rgb[2];
          */
         

        imageArray[i*3*img.cols+j*3]=img.at<unsigned char>(img.rows-1-i,3*j+2);
        imageArray[i*3*img.cols+j*3+1]=img.at<unsigned char>(img.rows-1-i,3*j+1);
        imageArray[i*3*img.cols+j*3+2]=img.at<unsigned char>(img.rows-1-i,3*j);;
      }
  }
}


int main(int argc,char* argv[])
{

    setup();

    thread render_loop;
    render_loop=thread(run);
    render_loop.join();

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


void drawAllFrames(vector<Pose> frames)
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

        glPushMatrix();
        glMultMatrixf(frames[i].Twc.data());
        drawFrame();
        glPopMatrix();            

    }
}

//====================== 线程相关 ======================
void setup(void)
{
     //========================= 窗口 ========================
    pangolin::CreateWindowAndBind(
        window_name,     //窗口标题
        640,        //窗口尺寸
        480);       //窗口尺寸
    glEnable(GL_DEPTH_TEST);

    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();
}


void run(void)
{

    // fetch the context and bind it to this thread

    Mat img=imread("../img/pic.jpg");
    Mat img2=imread("../img/test_pic.jpg");
    if(img.empty())
    {
        cout<<"Image is empty. Terminated."<<endl;
        return ;
    }
    else
    {
        cout<<"Image loaded complete."<<endl;
    }
    

    pangolin::BindToContext(window_name);

    vector<Pose> frames;


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
      .SetBounds(
          0.0, 
          1.0,
          0.0, 
          pangolin::Attach::Pix(UI_WIDTH));

    //接下来要开始准备添加控制选项了
    pangolin::Var<int> varTop("ui.Top",0,0,480);
    pangolin::Var<int> varBottom("ui.Bottom",480,0,480);
    pangolin::Var<int> varLeft("ui.Left",0,0,640);
    pangolin::Var<int> varRight("ui.Right",640,0,640);

    pangolin::Var<bool> checkRatio("ui.use_ratio",false,true);

    pangolin::Var<double> vardTop("ui.Top_r",0.0,0.0,1.0);
    pangolin::Var<double> vardBottom("ui.Bottom_r",1.0,0.0,1.0);
    pangolin::Var<double> vardLeft("ui.Left_r",0.0,0.0,1.0);
    pangolin::Var<double> vardRight("ui.Right_r",1.0,0.0,1.0);



   
    // =====================  图像窗口 ====================
    
    pangolin::View& d_image = pangolin::Display("image")
      .SetBounds(
          pangolin::Attach::Pix(0),
          pangolin::Attach::Pix(114),
          pangolin::Attach::Pix(396),
          pangolin::Attach::Pix(640),
          640.0/480)
      .SetLock(pangolin::LockCenter , pangolin::LockCenter);

    d_cam.AddDisplay(d_image);

    unsigned char* imageArray= new unsigned char[3*img.cols*img.rows];
    pangolin::GlTexture imageTexture(img.cols,img.rows,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);

    
    pangolin::View& d_image2 = pangolin::Display("image2")
      .SetBounds(
          0,
          0.2381,
          0.375,
          1.0,
          640.0/480)
      .SetLock(pangolin::LockCenter, pangolin::LockCenter);

    d_cam.AddDisplay(d_image2);

    unsigned char* imageArray2= new unsigned char[3*img2.cols*img2.rows];
    pangolin::GlTexture imageTexture2(img2.cols,img2.rows,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);

    

    while( !pangolin::ShouldQuit() )
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        d_cam.Activate(s_cam);

        //改变背景颜色
        glClearColor(0.3,0.3,0.3,0.0);

        /*
        if(checkRatio)
        {
            d_image.SetBounds(
                (double)vardTop,
                (double)vardBottom,
                (double)vardLeft,
                (double)vardRight);
        }
        else
        {
            d_image.SetBounds(
                pangolin::Attach::Pix((int)varTop),
                pangolin::Attach::Pix((int)varBottom),
                pangolin::Attach::Pix((int)varLeft),
                pangolin::Attach::Pix((int)varRight));

        }
    
        d_image2.SetBounds(
                (double)vardTop,
                (double)vardBottom,
                (double)vardLeft,
                (double)vardRight);
                
        */
        //Set some random image data and upload to GPU
        setImageData(imageArray,img);
        imageTexture.Upload(imageArray,GL_RGB,GL_UNSIGNED_BYTE);

    
        //Set some random image data and upload to GPU
        setImageData(imageArray2,img2);
        imageTexture2.Upload(imageArray2,GL_RGB,GL_UNSIGNED_BYTE);
    
        //display the image
        d_image.Activate();
        glColor3f(1.0,1.0,1.0);
        imageTexture.RenderToViewport();

   
        //display the image
        d_image2.Activate();
        glColor3f(1.0,1.0,1.0);
        imageTexture2.RenderToViewport();
   

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }
    
    //return 0;
    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();
}