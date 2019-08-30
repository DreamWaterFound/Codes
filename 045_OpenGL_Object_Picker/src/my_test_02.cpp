// 另外一篇博客提供的拾取的实现，但是目前还没有对其进行进一步的尝试


#include <stdio.h>
#include <math.h>
#include <GL/glut.h>
 
void drawTriangle(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2, GLfloat x3, GLfloat y3, GLfloat z)
{
	glBegin(GL_TRIANGLES);
		glVertex3f(x1, y1, z);
		glVertex3f(x2, y2, z);
		glVertex3f(x3, y3, z);
	glEnd();
}
void drawViewVolume(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2, GLfloat z1, GLfloat z2)
{
	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_LINE_LOOP);
		glVertex3f(x1, y1, -z1);
		glVertex3f(x2, y1, -z1);
		glVertex3f(x2, y2, -z1);
		glVertex3f(x1, y2, -z1);
	glEnd();
	glBegin(GL_LINE_LOOP);
		glVertex3f(x1, y1, -z2);
		glVertex3f(x2, y1, -z2);
		glVertex3f(x2, y2, -z2);
		glVertex3f(x1, y2, -z2);
	glEnd();
	glBegin(GL_LINES);
		glVertex3f(x1, y1, -z1);
		glVertex3f(x1, y1, -z2);
		glVertex3f(x1, y2, -z1);
		glVertex3f(x1, y2, -z2);
		glVertex3f(x2, y1, -z1);
		glVertex3f(x2, y1, -z2);
		glVertex3f(x2, y2, -z1);
		glVertex3f(x2, y2, -z2);
	glEnd();
}
void drawScene()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(40.0, 4.0 / 3.0, 1.0, 100.0);
 
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(7.5, 7.5, 12.5, 2.5, 2.5, -5.0, 0.0, 1.0, 0.0);
	glColor3f(0.0, 1.0, 0.0);
	drawTriangle(2.0, 2.0, 3.0, 2.0, 2.5, 3.0, -5.0);
	glColor3f(1.0, 0.0, 0.0);
	drawTriangle(2.0, 7.0, 3.0, 7.0, 2.5, 8.0, -5.0);
	glColor3f(1.0, 1.0, 0.0);
	drawTriangle(2.0, 2.0, 3.0, 2.0, 2.5, 3.0, 0.0);
	drawTriangle(2.0, 2.0, 3.0, 2.0, 2.5, 3.0, -10.0);
	drawViewVolume(0.0, 5.0, 0.0, 5.0, 0.0, 10.0);
}
void processHits(GLint hits, GLuint buffer[])
{
	unsigned int i, j;
	GLuint names, *ptr;
 
	printf("hits = %d \n", hits);
	ptr = (GLuint*)buffer;
	for (i = 0; i < hits; i++)
	{
		names = *ptr;
		printf(" number of names for hit = %d \n ", names);
		ptr++;
		printf(" z1 is %g;", (float)*ptr / 0x7fffffff);
		ptr++;
		printf(" z2 is %g \n", (float)*ptr / 0x7fffffff);
		ptr++;
		printf(" the name is ");
		for (j = 0; j < names; j++)
		{
			printf("%d ", *ptr);
			ptr++;
		}
		printf("\n ");
	}
}
 
#define BUFSIZE 512
 
void selectObjects()
{
	GLuint selectBuf[BUFSIZE];
	GLint hits;
 
	glSelectBuffer(BUFSIZE, selectBuf);
	(void)glRenderMode(GL_SELECT);
 
	glInitNames();
	glPushName(0);
 
	glPushMatrix();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 5.0, 0.0, 5.0, 0.0, 10.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glLoadName(1);
	drawTriangle(2.0, 2.0, 3.0, 2.0, 2.5, 3.0, -5.0);
	glLoadName(2);
	drawTriangle(2.0, 7.0, 3.0, 7.0, 2.5, 8.0, -5.0);
	glLoadName(3);
	drawTriangle(2.0, 2.0, 3.0, 2.0, 2.5, 3.0, 0.0);
	drawTriangle(2.0, 2.0, 3.0, 2.0, 2.5, 3.0, -10.0);
	glPopMatrix();
	glFlush();
 
	hits = glRenderMode(GL_RENDER);
	processHits(hits, selectBuf);
}
void init()
{
	glEnable(GL_DEPTH_TEST);
	glShadeModel(GL_FLAT);
}
void display()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	drawScene();
	selectObjects();
	glFlush();
}
void reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 5.0, 0.0, 5.0, 0.0, 10.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
void keyboard(unsigned char key, int x, int y)
{
	switch(key)
	{
	case 'c':
	case 'C':
		glutPostRedisplay();
		break;
	case 27:
		exit(0);
		break;
	default:
		break;
	}
}
int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE | GLUT_DEPTH);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(200, 200);
	glutCreateWindow(argv[0]);
	init();
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutDisplayFunc(display);
	glutMainLoop();
	return 0;
}