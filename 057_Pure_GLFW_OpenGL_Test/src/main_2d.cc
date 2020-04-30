#include <iostream>
#include <string>
// GLEW
// #define GLEW_STATIC
// #include <GL/glew.h>

#include <GL/gl3w.h>

// GLFW
#include <GLFW/glfw3.h>

#define GLT_ATTRIBUTE_VERTEX 1

 
using namespace std;

GLuint InitTraigleBatch(GLuint& hVAO, GLuint& hVBO)
{
    // 原始的三角形数据
    GLfloat vVerts[] = { -0.5f, 0.0f, 0.0f, 
		                  0.5f, 0.0f, 0.0f,
						  0.0f, 0.5f, 0.0f };
    // 3 个点                          
    GLuint nNumVerts = 3;

    // 创建 VAO 对象
    GLuint hVertexArrayObject;
    glGenVertexArrays(1, &hVertexArrayObject);
	glBindVertexArray(hVertexArrayObject);
    
    // 创建缓冲区, 并且进行绑定, 存储顶点坐标信息, 充当 VBO
    GLuint hVertexArray;
    glGenBuffers(1, &hVertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, hVertexArray);
    // TODO 缓冲区的那一章会有介绍
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 3 * nNumVerts, vVerts, GL_DYNAMIC_DRAW);

	// Set up the vertex array object -- 意义不明
	// glBindVertexArray(vertexArrayObject);
	
    // 这里是不是没有必要unmap之后又重新绑定
    glEnableVertexAttribArray(GLT_ATTRIBUTE_VERTEX);
    // glBindBuffer(GL_ARRAY_BUFFER, hVertexArray);
    glVertexAttribPointer(GLT_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, 0);

    // 不再使用之前的 VAO 对象
    glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindVertexArray(0);

    hVAO = hVertexArrayObject;
    hVBO = hVertexArray;

	return hVertexArrayObject;
}

// 绘制顶点batch
void DrawTraigles(GLuint nPrimitiveType, GLuint hVAO, GLuint hVBO)
{
    GLuint nNumVerts = 3;

	glBindVertexArray(hVAO);

    glEnableVertexAttribArray(GLT_ATTRIBUTE_VERTEX);
    // glBindBuffer(GL_ARRAY_BUFFER, hVBO);
    // glVertexAttribPointer(GLT_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glDrawArrays(nPrimitiveType, 0, nNumVerts);
    glDisableVertexAttribArray(GLT_ATTRIBUTE_VERTEX);

	glBindVertexArray(0);
}

// 加载着色器并构建渲染管线
GLuint InitShaders(void)
{
    // Temporary Shader objects
    GLuint hVertexShader;       // 顶点着色器句柄
    GLuint hFragmentShader;     // 片段着色器句柄
    GLuint hReturn = 0;         // 构建好的着色器程序
    GLint testVal;              // 用于检测着色器程序是否编译成功

    // 单位着色器程序
    const char *szIdentityShaderVP = "attribute vec4 vVertex;"
                                     "void main(void) "
                                     "{ gl_Position = vVertex; "
                                     "}";
                                
    const char *szIdentityShaderFP = "uniform vec4 vColor;"
									 "void main(void) "
									"{ gl_FragColor = vColor;"
									"}";
	
    // Create shader objects -- 创建着色器对象
    hVertexShader   = glCreateShader(GL_VERTEX_SHADER);
    hFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	
    // Load them. 
    // gltLoadShaderSrc(szIdentityShaderVP, hVertexShader);
    {
        GLchar *fsStringPtr[1];
        fsStringPtr[0] = (GLchar *)szIdentityShaderVP;
        glShaderSource(hVertexShader, 1, (const GLchar **)fsStringPtr, NULL);
    }

    {
        GLchar *fsStringPtr[1];
        fsStringPtr[0] = (GLchar *)szIdentityShaderFP;
        glShaderSource(hFragmentShader, 1, (const GLchar **)fsStringPtr, NULL);
    }
    // glShaderSource(hFragmentShader, 1, (const GLchar *)szIdentityShaderFP, NULL);

   
    // Compile them
    glCompileShader(hVertexShader);
    glCompileShader(hFragmentShader);
    
    // Check for errors
    glGetShaderiv(hVertexShader, GL_COMPILE_STATUS, &testVal);
    if(testVal == GL_FALSE)
    {
        char infoLog[1024];
        glGetShaderInfoLog(hVertexShader, 1024, NULL, infoLog);
        cout << "The shader at " << endl << szIdentityShaderVP << endl;
        cout << "failed to compile with the following error:" <<endl;
        cout << infoLog << endl;
        glDeleteShader(hVertexShader);
        glDeleteShader(hFragmentShader);
        return (GLuint)NULL;
    }
    
    glGetShaderiv(hFragmentShader, GL_COMPILE_STATUS, &testVal);
    if(testVal == GL_FALSE)
    {
        char infoLog[1024];
        glGetShaderInfoLog(hFragmentShader, 1024, NULL, infoLog);
        cout << "The shader at " << endl << szIdentityShaderFP << endl;
        cout << "failed to compile with the following error:" <<endl;
        cout << infoLog << endl;
        glDeleteShader(hVertexShader);
        glDeleteShader(hFragmentShader);
        return (GLuint)NULL;
    }
    
    // Link them - assuming it works...
    hReturn = glCreateProgram();
    glAttachShader(hReturn, hVertexShader);
    glAttachShader(hReturn, hFragmentShader);

    // 添加属性信息
    glBindAttribLocation(hReturn, GLT_ATTRIBUTE_VERTEX,"vVertex");

    glLinkProgram(hReturn);
	
    // These are no longer needed
    glDeleteShader(hVertexShader);
    glDeleteShader(hFragmentShader);  
    
    // Make sure link worked too
    glGetProgramiv(hReturn, GL_LINK_STATUS, &testVal);
    if(testVal == GL_FALSE)
    {
		glDeleteProgram(hReturn);
        cout << "failed to generate a shader program." << endl;
		return (GLuint)NULL;
    }
    
    return hReturn;  
}

// 启用之前生成的着色器对象
void EnableShaders(GLuint hProgram, GLfloat* fColors)
{
    // 使用指定的着色器程序
	glUseProgram(hProgram);
    // 设置统一值
    GLint iColor = glGetUniformLocation(hProgram, "vColor");
    // vColor = va_arg(uniformList, M3DVector4f*);
    glUniform4fv(iColor, 1, fColors);
}

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
                                    
    cout << "Window resized [" << width << " , " << height << "] !" << endl;
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
            cout << "ESC Pressed." <<endl;
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

    auto window = glfwCreateWindow(640, 480, "Pure OpenGL Test", nullptr, nullptr);
    if(window == nullptr) {
        return false;
    }

    // 设置各种事件的回调函数
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);

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

    // 初始化着色器
    GLuint hShaderProgram = InitShaders();
    if(hShaderProgram == GLuint(NULL))
    {
        cout << "init shaders failed." << endl;
        glfwDestroyWindow(window);
        glfwTerminate();
    }
    else
    {
        cout << "init shaders OK." << endl;
    }
    
    // 初始化要绘制的图像
    GLuint hVertexArray;
    GLuint hVertexArrayObject;
    InitTraigleBatch(hVertexArrayObject, hVertexArray);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // 处理系统信息(键鼠输入, 其他窗口交互信息等)
        glfwPollEvents();

        // processInput(window);

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        // 这里就是对背景进行绘图的地方

        GLfloat afColor[] = {1.0f, 0.0f, 0.0f, 1.0f};
        EnableShaders(hShaderProgram, afColor);

        DrawTraigles(GL_TRIANGLES, hVertexArrayObject, hVertexArray);

        glfwSwapBuffers(window);
    }

    // 注意需要删除着色器程序
    glDeleteProgram(hShaderProgram);
    // 删除顶点
    glDeleteVertexArrays(1, &hVertexArrayObject);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}