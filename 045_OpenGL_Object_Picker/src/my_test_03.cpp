// 基于my_test_01.cpp
// 测试多个物体的拾取,实测400个物体是没有问题的

#include <GL/glut.h>
#include <iostream>
#include <cmath>

using namespace std;

int mouseX=0,mouseY=0;
bool isClicked=false;

void draw(GLenum model=GL_RENDER);    


void SelectObject(GLint x, GLint y)    
{    
    GLuint selectBuff[32]={0};//创建一个保存选择结果的数组   
    GLint hits, viewport[4];      
    glGetIntegerv(GL_VIEWPORT, viewport); //获得viewport    
    glSelectBuffer(32, selectBuff); //告诉OpenGL初始化  selectbuffer    

    //进入选择模式    
    glRenderMode(GL_SELECT);   
    glInitNames();  //初始化名字栈    
    // glPushName(0);  //在名字栈中放入一个初始化名字，这里为‘0’    

    glMatrixMode(GL_PROJECTION);    //进入投影阶段准备拾取    
    // glMatrixMode(GL_MODELVIEW);    //进入投影阶段准备拾取    

    glPushMatrix();     //保存以前的投影矩阵    
    glLoadIdentity();   //载入单位矩阵    



    gluPickMatrix(  x,           // 设定我们选择框的大小，建立拾取矩阵，就是上面的公式  
                    viewport[3]-y,    // viewport[3]保存的是窗口的高度，窗口坐标转换为OpenGL坐标（OPengl窗口坐标系）   
                    2,2,              // 选择框的大小为2，2    
                    viewport          // 视口信息，包括视口的起始位置和大小    
                    );        

    //投影处理，并归一化处理
    glOrtho(-25, 25, -25, 25, -25, 25);     //拾取矩阵乘以投影矩阵，这样就可以让选择框放大为和视体一样大    

    draw(GL_SELECT);    // 该函数中渲染物体，并且给物体设定名字    

    glMatrixMode(GL_PROJECTION);    

    glPopMatrix();  // 返回正常的投影变换    

    hits = glRenderMode(GL_RENDER); // 从选择模式返回正常模式,该函数返回选择到对象的个数    

    if(hits > 0)    
    {
        cout<<"Hits: "<<hits<<endl;
        for(size_t i=0;i<hits;++i)
        {
            cout<<"\tSelected Obj Number:"<<selectBuff[4*i+0]<<endl;
            cout<<"\tMin depth:"<<1.0*selectBuff[4*i+1]/0xffffffff<<endl;
            cout<<"\tMax depth:"<<1.0*selectBuff[4*i+2]/0xffffffff<<endl;
            cout<<"\tSelected Obj Name:"<<selectBuff[4*i+3]<<endl;
            cout<<"======"<<endl;
        }
        cout<<endl;
    }
    else
    {
        cout<<"No Body selected."<<endl;
    }
}    
  


void draw(GLenum model)    
{    
    if(model==GL_SELECT)    
    {    
        
        const int rows=10;
        const int cols=10;

        size_t cnt=1;
        for(int i=-rows;i<rows;++i)
        {
            for(int j=-cols;j<cols;++j)
            {   
                // 红色
                switch(((int)fabs(i)*cols+(int)fabs(j))%3)
                {
                    case 0:
                        glColor3f(0.8,0.0,0.0);    
                        break;
                    case 1:
                        glColor3f(0.0,0.8,0.0);
                        break;
                    case 2:
                        glColor3f(0.0,0.0,0.8);
                        break;
                }

                glPushName(cnt++);
                glPushMatrix();    
                // 平移到新位置
                glTranslatef(2*i, 2*j, 0);    
                glBegin(GL_QUADS);    
                glVertex3f(-1, -1, 0);    
                glVertex3f( 1, -1, 0);    
                glVertex3f( 1, 1, 0);    
                glVertex3f(-1, 1, 0);    
                glEnd();    
                glPopMatrix();
                glPopName();
            }
        }

        /*
        glColor3f(1.0,0.0,0.0);    
        glLoadName(100);  //第一个矩形命名
        glPushMatrix();    
        glTranslatef(-5, 0.0, 10.0);    
        glBegin(GL_QUADS);    
        glVertex3f(-1, -1, 0);    
        glVertex3f( 1, -1, 0);    
        glVertex3f( 1, 1, 0);    
        glVertex3f(-1, 1, 0);    
        glEnd();    
        glPopMatrix();    

        glColor3f(0.0,0.0,1.0);    
        glLoadName(101); //第二个矩形命名
        glPushMatrix();    
        glTranslatef(5, 0.0, -10.0);    
        glBegin(GL_QUADS);    
        glVertex3f(-1, -1, 0);   
        glVertex3f( 1, -1, 0);    
        glVertex3f( 1, 1, 0);    
        glVertex3f(-1, 1, 0);    
        glEnd();    
        glPopMatrix(); 
        */   
        
    }    
    else //正常渲染  
    {    
        const int rows=10;
        const int cols=10;
        for(int i=-rows;i<rows;++i)
        {
            for(int j=-cols;j<cols;++j)
            {   
                // 红色
                switch(((int)fabs(i)*cols+(int)fabs(j))%3)
                {
                    case 0:
                        glColor3f(0.8,0.0,0.0);    
                        break;
                    case 1:
                        glColor3f(0.0,0.8,0.0);
                        break;
                    case 2:
                        glColor3f(0.0,0.0,0.8);
                        break;
                }

                glPushMatrix();    
                // 平移到新位置
                glTranslatef(2*i, 2*j, 0);    
                glBegin(GL_QUADS);    
                glVertex3f(-1, -1, 0);    
                glVertex3f( 1, -1, 0);    
                glVertex3f( 1, 1, 0);    
                glVertex3f(-1, 1, 0);    
                glEnd();    
                glPopMatrix(); 
            }
        }
    }    

    

}

// 显示循环的过程中调用的函数;如果发生右键菜单的调用,那么这里也会被重绘
void display()
{
     // 清空画布为背景颜色
    glClearColor(0.3,0.3,0.3,1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // gluLookAt(1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0);
   
    draw();

    glutSwapBuffers();
}

// 当窗口发生reshape事件时调用的函数
void reshape(int w,int h)
{
    // 重设视口(view port)大小
    glViewport(0,0,w,h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-25.0,25.0,-25.0,25.0,-25.0,25.0);
    cout<<"Reshape called."<<endl;
}

// 初始化场景
void init()
{
    // 清空画布为背景颜色
    glClearColor(0.3,0.3,0.3,1.0);
    // 设置当前画笔颜色
    glColor3f(0.0,0.0,0.0);
}

void OnKeyBoards(unsigned char key,int x,int y)
{
    if(key==27) exit(0);
    cout<<"key="<<(int)key<<endl;
}

void OnMouse(int button,int state,int x,int y)
{
    
    if(button==GLUT_LEFT_BUTTON && state==GLUT_DOWN)
    {
        cout<<"Click at ("<<x<<","<<y<<")"<<endl;
        // isClicked=true;
        SelectObject(x,y);

    }
}

// 右键菜单
void mymenu(int value){
	if (value == 1){
		cout<<"menu #1 selected."<<endl;
	}
	if (value == 2){
		exit(0);
	}
}


// 主函数 
int main(int argc,char** argv)
{
    // 初始化交互界面
    glutInit(&argc,argv);
    // 显示模式
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    // 窗口大小
    glutInitWindowSize(800,800);
    // 窗口显示的位置
    glutInitWindowPosition(100,100);
    // 窗口标题
    glutCreateWindow("Cube");
    // 当窗口发生reshape事件时的响应函数
    glutReshapeFunc(reshape);
    // 键盘响应
    glutKeyboardFunc(OnKeyBoards);
    // 鼠标响应
    glutMouseFunc(OnMouse);
    // 创建右键菜单
	glutCreateMenu(mymenu);
    // 添加菜单项
    glutAddMenuEntry("Clear Screen",1);//添加菜单项
    glutAddMenuEntry("Exit",2);
    glutAttachMenu(GLUT_RIGHT_BUTTON);//把当前菜单注册到指定的鼠标键
    // 绘制函数
    glutDisplayFunc(display);
    // 调用自己写的初始化函数
    init();


    const GLubyte * name = glGetString(GL_VENDOR);
    const GLubyte * biaoshifu = glGetString(GL_RENDERER);
    const GLubyte * OpenGLVersion = glGetString(GL_VERSION);
    const GLubyte * gluVersion = gluGetString(GLU_VERSION);

    cout<<"GL_VENDOR:"<<(char*)name<<endl;
    cout<<"GL_RENDERER:"<<(char*)biaoshifu<<endl;
    cout<<"GL_VERSION:"<<(char*)OpenGLVersion<<endl;
    cout<<"GLU_VERSION:"<<(char*)gluVersion<<endl;
    


    // 进入显示循环
    glutMainLoop();
    return 0;
}