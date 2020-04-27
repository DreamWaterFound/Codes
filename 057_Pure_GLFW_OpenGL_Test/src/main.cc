#include <iostream>
#include <string>
// GLEW
// #define GLEW_STATIC
// #include <GL/glew.h>

#include <GL/gl3w.h>

// GLFW
#include <GLFW/glfw3.h>

 
using namespace std;

// 主动按键处理
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)//获取按键，如果等于esc 
        glfwSetWindowShouldClose(window, true);//利用强制窗口应该关闭
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)//这个是窗口变化的回调函数。。注意输入参数
                                                                         //是一个glfw的窗口，一个宽度和高度
{
    glViewport(0, 0, width, height);//这个是回调函数内的内容
                                    //这里是将视口改成变化后的的窗口大小
                                    //注意需要的注册该回调函数
                                    //glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
                                    //两个参数是，glfw的窗口以及回调函数
                                    
    cout << "Window resized!" <<endl;
}

//按键回调函数
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action != GLFW_PRESS)
		return;
	switch (key)
		{
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		default:
			break;
		}
}
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (action == GLFW_PRESS) switch(button)
        {
        case GLFW_MOUSE_BUTTON_LEFT:
            cout << "Mosue left button clicked!" << endl;
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            cout << "Mosue middle button clicked!" << endl;
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            cout << "Mosue right button clicked!" <<endl;
            break;
        default:
            return;
        }
	return;
}
void cursor_position_callback(GLFWwindow* window, double x, double y)
{
    cout << "Mouse position move to [" << int(x) << " : " << int(y) << "]" <<endl;
	return;
}
void scroll_callback(GLFWwindow* window, double x, double y)
{
    cout << "Mouse position scroll at [" << int(x) << " : " << int(y) << "]" <<endl;
	return;
}

int main(void)
{

    glfwSetErrorCallback([](int err, const char* desc) { std::cerr << "glfw error " << err << ": " << desc << std::endl; });
    if(!glfwInit()) {
        std::cerr << "failed to initialize GLFW" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);//设置主版本号
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);//设置次版本号
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    auto window = glfwCreateWindow(1024, 728, "Pure OpenGL Test", nullptr, nullptr);
    if(window == nullptr) {
        return false;
    }

    // 设置各种事件的回调函数
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetScrollCallback(window, scroll_callback);


    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

   
    if (gl3wInit()) {
                fprintf(stderr, "failed to initialize OpenGL\n");
                return -1;
        }
        if (!gl3wIsSupported(3, 2)) {
                fprintf(stderr, "OpenGL 3.2 not supported\n");
                return -1;
        }
        printf("OpenGL %s, GLSL %s\n", glGetString(GL_VERSION),
               glGetString(GL_SHADING_LANGUAGE_VERSION));


    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // 处理系统信息(键鼠输入, 其他窗口交互信息等)
        glfwPollEvents();

        processInput(window);

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        // 这里就是对背景进行绘图的地方

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}