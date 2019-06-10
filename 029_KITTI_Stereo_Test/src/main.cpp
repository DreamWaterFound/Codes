#include <iostream>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <vector>


using namespace std;
using namespace cv;

// 相机内参
#define L_FX 7.188560000000e+02
#define L_FY 7.188560000000e+02
#define L_CX 6.071928000000e+02
#define L_CY 1.852157000000e+02

#define R_FX 7.188560000000e+02
#define R_FY 7.188560000000e+02
#define R_CX 6.071928000000e+02
#define R_CY 1.852157000000e+02

#define FB   3.861448000000e+02

#define IMG_W 1024.0
#define IMG_H 768.0




void prefilterXSobel(const cv::Mat& src, cv::Mat& dst, int ftzero);

template <typename T> void filterSpecklesImpl(cv::Mat& img, int newVal, int maxSpeckleSize, int maxDiff, cv::Mat& _buf);

void initColor(void);


typedef struct _Color
{
//public:
    unsigned char r;
    unsigned char g;
    unsigned char b;

    /*
    _Color(unsigned char _r,unsigned char _g,unsigned char _b)
    {    r=_r;g=_g;b=_b;  }
    */

}Color;

vector<Color> HueCircle;


int main(int argc, char* argv[])
{
    cout<<"Kitti Streo Test."<<endl;
    cout<<"Complied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    if(argc!=4)
    {
        cout<<"Usage: "<<argv[0]<<" img_left img_right img_dis"<<endl;
        return 1;
    }

    // 读入左右双目图像
    Mat imgLeft =imread(argv[1],IMREAD_GRAYSCALE);
    Mat imgRight=imread(argv[2],IMREAD_GRAYSCALE);

    if(imgLeft.empty())
    {
        cout<<"Error: img_left "<<argv[1]<<" is empty!"<<endl;
        return 2;
    }

    if(imgRight.empty())
    {
        cout<<"Error: img_left "<<argv[1]<<" is empty!"<<endl;
        return 2;
    }

    imshow("img_left",imgLeft);
    imshow("img_right",imgRight);

    waitKey(100);

    Mat imgLefted,imgRighted;

    

    // 最小视差
    int mindisparity = 0;
    // 视差搜索范围长度
	int ndisparities = 64;  
    // SAD代价计算窗口大小
	int SADWindowSize = 11; 
	//SGBM
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);

    // 能量函数参数
	int P1 = 8 * imgLeft.channels() * SADWindowSize* SADWindowSize;
    // 能量函数参数
	int P2 = 32 * imgRight.channels() * SADWindowSize* SADWindowSize;

    // 下面就是各种配置了
	sgbm->setP1(P1);
	sgbm->setP2(P2);
	sgbm->setPreFilterCap(15);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleRange(2);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setDisp12MaxDiff(1);
	//sgbm->setMode(cv::StereoSGBM::MODE_HH);

    // 对原始图像进行预处理
    // imgLeft.copyTo(imgLefted);
    // imgRight.copyTo(imgRighted);
    // prefilterXSobel(imgLeft, imgLefted, sgbm->getPreFilterCap());
    // prefilterXSobel(imgRight, imgRighted, sgbm->getPreFilterCap());

    // imshow("img_left",imgLefted);
    // imshow("img_right",imgRighted);

    // waitKey(0);

    // 
    Mat disp;
	sgbm->compute(imgLeft, imgRight, disp);
	disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值
	Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示
	normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);

    // display
    imshow("dis",disp8U);

    waitKey(100);

	imwrite(argv[3], disp8U);

    // 生成相机内参数矩阵
    Eigen::Matrix3d K;
    K<<L_FX,    0.0,    L_CX,
       0.0,     L_FY,   L_CY,
       0.0,     0.0,    1.0;
    Eigen::Matrix3d K_inv=K.inverse();

    // 准备深度着色
    double min_d,max_d;
    int max_id[2],min_id[2];
    minMaxIdx(disp,&min_d,&max_d,min_id,max_id);

    min_d=min_d<5 ? 5:min_d;

    double max_z=FB/min_d,min_z=FB/max_d;

    double factor=1.0*1536/(max_z-min_z);


    initColor();


    // ================================== 准备可视化 =================================
    pangolin::CreateWindowAndBind(
        "PointClouds",     //窗口标题
        IMG_W,        //窗口尺寸
        IMG_H);       //窗口尺寸
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(
            IMG_W,IMG_H,            //相机图像的长和宽
            L_FX,L_FY,L_CX,L_CY,    //相机的内参,fu fv u0 v0
            0.2,1000),           //相机所能够看到的最浅和最深的像素
        pangolin::ModelViewLookAt(
            -2,2,-2,            //相机光心位置,NOTICE z轴不要设置为0
            0,0,0,              //相机要看的位置
            pangolin::AxisY)    //和观察的方向有关
    );

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(
                0.0, 1.0, 0.0, 1.0,     //表示整个窗口都可以观测到
                -IMG_W/IMG_H)         //窗口的比例
            .SetHandler(&handler);

    while( !pangolin::ShouldQuit() )
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        glClearColor(1,1,1,0.0);

        // Render OpenGL Cube
        //pangolin::glDrawColouredCube();

        //尝试按照谢晓佳的视频中给出的代码绘制
        // pangolin::glDrawAxis(3);

        glPointSize(1.0f);
        glBegin(GL_POINTS);
        

        // 绘制当前帧点云
        for(int x=0;x<disp.rows;++x)
        {
            for(int y=0;y<disp.cols;++y)
            {
                float d=disp.at<float>(x,y);

                if(d>=min_d)
                {
                    double z=FB/d;
                    
                    Eigen::Vector3d position=z*K_inv*Eigen::Vector3d(y,x,1);
                    // cout<<"Debug: disp8U.at<uint8_t>(x,y)="<<(int)(disp8U.at<uint8_t>(x,y))<<endl;
                    // Color c=HueCircle[(int)(disp8U.at<char>(x,y))];
                    // if((size_t)(z*1000)<1535)
                    {
                        // cout<<"debug: z="<<z<<endl;
                        // Color c=HueCircle[(size_t)(z*10)];
                        // cout<<"debug: r="<<(int)c.r<<"\tg="<<(int)c.g<<"\tb="<<(int)c.b<<"\t(size_t)(z*10)="<<(size_t)(z*10)<<endl;
                        size_t index=(z-min_z)*factor;
                        Color c=HueCircle[index];


                        // 画点
                        glColor3f(0,0,0);
                        // glColor3f(c.r/255.0,c.g/255.0,c.b/255.0);
                        // glColor3f(c.r,c.g,c.b);
                        glVertex3f(-position[0],-position[1],position[2]);
                    }
                    
                }
                
            }
        }

        glEnd();
        //不要忘记了这个东西!!!
        glFlush();


        // Swap frames and Process Events
        pangolin::FinishFrame();

        waitKey(1);

    }
    

    return 0;
}


void initColor(void)
{
    

     //颜色环初始化
    HueCircle.reserve(1536);
    HueCircle.resize(1536);

    for (int i = 0;i < 255;i++)
	{
		HueCircle[i].r = 255;
		HueCircle[i].g = i;
		HueCircle[i].b = 0;

		HueCircle[i+255].r = 255-i;
		HueCircle[i+255].g = 255;
		HueCircle[i+255].b = 0;

		HueCircle[i+511].r = 0;
		HueCircle[i+511].g = 255;
		HueCircle[i+511].b = i;

		HueCircle[i+767].r = 0;
		HueCircle[i+767].g = 255-i;
		HueCircle[i+767].b = 255;

		HueCircle[i+1023].r = i;
		HueCircle[i+1023].g = 0;
		HueCircle[i+1023].b = 255;

		HueCircle[i+1279].r = 255;
		HueCircle[i+1279].g = 0;
		HueCircle[i+1279].b = 255-i;
	}

	HueCircle[1534].r = 0;
	HueCircle[1534].g = 0;
	HueCircle[1534].b = 0;

	HueCircle[1535].r = 255;
	HueCircle[1535].g = 255;
	HueCircle[1535].b = 255;

    // while(1)
    // {
    //     int index;
    //     cout<<"Index? ";
    //     cin>>index;
    //     cout<<"\tr="<<()HueCircle[index].r
    //         <<"\tg="<<HueCircle[index].g
    //         <<"\tb="<<HueCircle[index].b<<endl;
    // }

 


  


}