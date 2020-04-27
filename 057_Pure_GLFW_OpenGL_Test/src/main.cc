#include <iostream>
#include <string>
// GLEW
// #define GLEW_STATIC
// #include <GL/glew.h>
// GLFW
#include <GLFW/glfw3.h>
 
using namespace std;
 
int main(void)
{
 
	// 初始化GLFW  
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}
 
	//GLFW
	glfwInit();
 
 
	//创建窗口
	GLFWwindow* window = glfwCreateWindow(800, 600, "Pure OpenGL Graphic Demo", nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	/*循环绘制 */
	while (!glfwWindowShouldClose(window))
	{
		//清屏背景
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

       
 
		glfwSwapBuffers(window);
 
		/* Poll for and process events */
		glfwPollEvents();


	}
 
	glfwTerminate();
	return 0;
}