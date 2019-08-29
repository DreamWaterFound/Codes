#include <GL/glut.h>


void SelectObject(GLint x, GLint y)    
  
{    
  GLuint selectBuff[32]={0};//创建一个保存选择结果的数组    
  GLint hits, viewport[4];      
  
  glGetIntegerv(GL_VIEWPORT, viewport); //获得viewport    
  glSelectBuffer(64, selectBuff); //告诉OpenGL初始化  selectbuffer    

   //进入选择模式    
  glRenderMode(GL_SELECT);   
  
  glInitNames();  //初始化名字栈    
  glPushName(0);  //在名字栈中放入一个初始化名字，这里为‘0’    
  
  glMatrixMode(GL_PROJECTION);    //进入投影阶段准备拾取    
  
  glPushMatrix();     //保存以前的投影矩阵    
  glLoadIdentity();   //载入单位矩阵    
  
  float m[16];    
  glGetFloatv(GL_PROJECTION_MATRIX, m);  //监控当前的投影矩阵 
  
  gluPickMatrix( x,           // 设定我们选择框的大小，建立拾取矩阵，就是上面的公式  
   viewport[3]-y,    // viewport[3]保存的是窗口的高度，窗口坐标转换为OpenGL坐标（OPengl窗口坐标系）   
   2,2,              // 选择框的大小为2，2    
   viewport          // 视口信息，包括视口的起始位置和大小    
   );        
  
    glGetFloatv(GL_PROJECTION_MATRIX, m);//查看当前的拾取矩阵  
    //投影处理，并归一化处理
    glOrtho(-10, 10, -10, 10, -10, 10);     //拾取矩阵乘以投影矩阵，这样就可以让选择框放大为和视体一样大    
    glGetFloatv(GL_PROJECTION_MATRIX, m);    
  
    draw(GL_SELECT);    // 该函数中渲染物体，并且给物体设定名字    
  
    glMatrixMode(GL_PROJECTION);    

    glPopMatrix();  // 返回正常的投影变换    
  


    glGetFloatv(GL_PROJECTION_MATRIX, m);//即还原在选择操作之前的投影变换矩阵 
  
    hits = glRenderMode(GL_RENDER); // 从选择模式返回正常模式,该函数返回选择到对象的个数    
  
    if(hits > 0)    
  
        // processSelect(selectBuff);  //  选择结果处理    
  
}    
  


    void draw(GLenum model=GL_RENDER)    
  
    {    
         if(model==GL_SELECT)    
        {    
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
  
    }    
     else //正常渲染  
    {    
        glColor3f(1.0,0.0,0.0);    
        glPushMatrix();    
            glTranslatef(-5, 0.0, -5.0);    
            glBegin(GL_QUADS);    
            glVertex3f(-1, -1, 0);    
            glVertex3f( 1, -1, 0);    
            glVertex3f( 1, 1, 0);    
            glVertex3f(-1, 1, 0);    
        glEnd();    
        glPopMatrix();    
  
        
  
       glColor3f(0.0,0.0,1.0);    
       glPushMatrix();   
       glTranslatef(5, 0.0, -10.0);   
       glBegin(GL_QUADS);    
       glVertex3f(-1, -1, 0);    
            glVertex3f( 1, -1, 0);    
            glVertex3f( 1, 1, 0);    
            glVertex3f(-1, 1, 0);    
            glEnd();    
        glPopMatrix();    
   }    
  
}

void display()

{

 glClear(GL_COLOR_BUFFER_BIT);

 glMatrixMode(GL_MODELVIEW);

 glLoadIdentity();

 gluLookAt(1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0);

 glutWireCube(3.0);

 draw();

 glutSwapBuffers();

}

void reshape(int w,int h)

{

 glViewport(0,0,w,h);

 glMatrixMode(GL_PROJECTION);

 glLoadIdentity();

 glOrtho(-40.0,40.0,-40.0,40.0,-40.0,40.0);

}

void init()

{

 glClearColor(1.0,1.0,1.0,1.0);

 glColor3f(0.0,0.0,0.0);

}

int main(int argc,char** argv)

{

 glutInit(&argc,argv);

 glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

 glutInitWindowSize(500,500);

 glutInitWindowPosition(100,100);

 glutCreateWindow("Cube");

 glutReshapeFunc(reshape);

 glutDisplayFunc(display);

 init();

 glutMainLoop();

 return 0;
}