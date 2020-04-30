#include <iostream>
#include <string>
#include <cmath>
// gl3w
#include <GL/gl3w.h>

// GLFW
#include <GLFW/glfw3.h>

#include <cstring>

#define M3D_PI (3.141562954f)

#define GLT_ATTRIBUTE_VERTEX 1
 
using namespace std;

float gfpProjectMatrix[16];
float gfpModelViewMatrix[16];
float gfpCameraFrame[16];
float gfpObjectFrame[16];
float gfpTmp1[16], gfpTmp2[16], gfpTmp3[16], gfpTmp4[16];

// ========== 函数原型 ===============
void GenerateProjectionMatrix(float fFov, float fAspect, float fNear, float fFar, float* fProjectMatrix);
void m3dLoadIdentity44(float* m);

void GetCameraFrame(float* pfCameraFrame);
void GetObjectFrame(float* pfObjectFrame);

void m3dCrossProduct3(float* result, const float* u, const float* v);
void m3dTranslationMatrix44(float* m, float x, float y, float z);
void m3dMatrixMultiply44(float* product, const float* a, const float* b );
void m3dSetMatrixColumn44(float* dst, const float* src, const int column);

void DisplayMatrix(float* mat);

// =======================

GLuint InitTraigleBatch(GLuint& hVAO, GLuint& hVBO)
{
    // 原始的三棱锥数据
    GLfloat vVerts[12][3] = {   -2.0f, 0.0f, -2.0f, 
                                2.0f, 0.0f, -2.0f, 
                                0.0f, 4.0f, 0.0f,
                                
                                2.0f, 0.0f, -2.0f,
                                2.0f, 0.0f, 2.0f,
                                0.0f, 4.0f, 0.0f,
                                
                                2.0f, 0.0f, 2.0f,
                                -2.0f, 0.0f, 2.0f,
                                0.0f, 4.0f, 0.0f,
                                
                                -2.0f, 0.0f, 2.0f,
                                -2.0f, 0.0f, -2.0f,
                                0.0f, 4.0f, 0.0f};
    // 12 个点                          
    GLuint nNumVerts = 12;

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
    glEnableVertexAttribArray(GLT_ATTRIBUTE_VERTEX);
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
    GLuint nNumVerts = 12;

	glBindVertexArray(hVAO);

    glEnableVertexAttribArray(GLT_ATTRIBUTE_VERTEX);

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
    const char *szFlatShaderVP = "uniform mat4 mvpMatrix;"
                                 "attribute vec4 vVertex;"
                                 "void main(void) "
                                 "{ gl_Position = mvpMatrix * vVertex; "
                                 "}";
                                
    const char *szFlatShaderFP = "uniform vec4 vColor;"
								 "void main(void) "
								 "{ gl_FragColor = vColor; "
								 "}";
	
    // Create shader objects -- 创建着色器对象
    hVertexShader   = glCreateShader(GL_VERTEX_SHADER);
    hFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	
    // Load them. 
    {
        GLchar *fsStringPtr[1];
        fsStringPtr[0] = (GLchar *)szFlatShaderVP;
        glShaderSource(hVertexShader, 1, (const GLchar **)fsStringPtr, NULL);
    }

    {
        GLchar *fsStringPtr[1];
        fsStringPtr[0] = (GLchar *)szFlatShaderFP;
        glShaderSource(hFragmentShader, 1, (const GLchar **)fsStringPtr, NULL);
    }
   
    // Compile them
    glCompileShader(hVertexShader);
    glCompileShader(hFragmentShader);
    
    // Check for errors
    glGetShaderiv(hVertexShader, GL_COMPILE_STATUS, &testVal);
    if(testVal == GL_FALSE)
    {
        char infoLog[1024];
        glGetShaderInfoLog(hVertexShader, 1024, NULL, infoLog);
        cout << "The shader at " << endl << szFlatShaderVP << endl;
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
        cout << "The shader at " << endl << szFlatShaderFP << endl;
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
void EnableShaders(GLuint hProgram, GLfloat* fColors, GLfloat* mvpMatrix)
{
    // 使用指定的着色器程序
	glUseProgram(hProgram);
    // 设置统一值 -- 颜色
    GLint iColor = glGetUniformLocation(hProgram, "vColor");
    glUniform4fv(iColor, 1, fColors);
    // 设置统一值 -- 变换投影矩阵
    GLint iTransform = glGetUniformLocation(hProgram, "mvpMatrix");
    glUniformMatrix4fv(iTransform, 1, GL_FALSE, mvpMatrix);
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

    GenerateProjectionMatrix(35.0f, float(width) / float(height), 1.0f, 500.0f, gfpProjectMatrix);
                                    
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

	glEnable(GL_DEPTH_TEST);

     int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    framebuffer_size_callback(window, display_w, display_h);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // 处理系统信息(键鼠输入, 其他窗口交互信息等)
        glfwPollEvents();

        // processInput(window);

       
        GetCameraFrame(gfpCameraFrame);
        GetObjectFrame(gfpObjectFrame);

        m3dMatrixMultiply44(gfpModelViewMatrix, gfpCameraFrame, gfpObjectFrame);
        m3dMatrixMultiply44(gfpTmp1, gfpProjectMatrix, gfpModelViewMatrix);

        // 准备绘图
    	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        // 这里就是对背景进行绘图的地方

        // GLfloat afColorRed[]   = {1.0f, 0.0f, 0.0f, 1.0f};
        GLfloat afColorGreen[] = {0.0f, 1.0f, 0.0f, 1.0f};
        GLfloat afColorBlack[] = {0.0f, 0.0f, 0.0f, 1.0f};
        EnableShaders(hShaderProgram, afColorGreen, gfpTmp1);

        DrawTraigles(GL_TRIANGLES, hVertexArrayObject, hVertexArray);

        glPolygonOffset(-1.0f, -1.0f);      // Shift depth values

        glEnable(GL_POLYGON_OFFSET_LINE);

        // Draw lines antialiased
        // 启用抗锯齿效果
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        // Draw black wireframe version of geometry
        // 绘制线
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(2.5f);
        EnableShaders(hShaderProgram, afColorBlack, gfpTmp1);
        DrawTraigles(GL_TRIANGLES, hVertexArrayObject, hVertexArray);
        
        
        // Put everything back the way we found it
        // 恢复正常
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDisable(GL_POLYGON_OFFSET_LINE);
        glLineWidth(1.0f);
        glDisable(GL_BLEND);
        glDisable(GL_LINE_SMOOTH);

        glfwSwapBuffers(window);
    }

    // 注意需要删除着色器程序
    glDeleteProgram(hShaderProgram);
    // 删除顶点
    glDeleteVertexArrays(1, &hVertexArrayObject);
    // 删除缓冲区
    glDeleteBuffers(1, &hVertexArray);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

// 生成投影矩阵
void GenerateProjectionMatrix(float fFov, float fAspect, float fNear, float fFar, float* fpProjectMatrix)
{
    float xmin, xmax, ymin, ymax;       // Dimensions of near clipping plane
        // float xFmin, xFmax, yFmin, yFmax;   // Dimensions of far clipping plane

        // Do the Math for the near clipping plane
        ymax = fNear * float(tan( fFov * M3D_PI / 360.0 ));
        ymin = -ymax;
        xmin = ymin * fAspect;
        xmax = -xmin;
            
        // Construct the projection matrix
        m3dLoadIdentity44(fpProjectMatrix);
        // 看来这个是有单独的计算公式
        fpProjectMatrix[0]  = (2.0f * fNear)/(xmax - xmin);
        fpProjectMatrix[5]  = (2.0f * fNear)/(ymax - ymin);
        fpProjectMatrix[8]  = (xmax + xmin) / (xmax - xmin);
        fpProjectMatrix[9]  = (ymax + ymin) / (ymax - ymin);
        fpProjectMatrix[10] = -((fFar + fNear)/(fFar - fNear));
        fpProjectMatrix[11] = -1.0f;
        fpProjectMatrix[14] = -((2.0f * fFar * fNear)/(fFar - fNear));
        fpProjectMatrix[15] = 0.0f;
}

// 4x4 double
void m3dLoadIdentity44(float* m)
{
    m[ 0] = 1.0f;
    m[ 1] = 0.0f;
    m[ 2] = 0.0f;
    m[ 3] = 0.0f;

    m[ 4] = 0.0f;
    m[ 5] = 1.0f;
    m[ 6] = 0.0f;
    m[ 7] = 0.0f;

    m[ 8] = 0.0f;
    m[ 9] = 0.0f;
    m[10] = 1.0f;
    m[11] = 0.0f;

    m[12] = 0.0f;
    m[13] = 0.0f;
    m[14] = 0.0f;
    m[15] = 1.0f;
}

void GetCameraFrame(float* m)
{
    float vOrigin[3], vUp[3], vForward[3];

    // At origin
    vOrigin[0] = 0.0f; vOrigin[1] = 0.0f; vOrigin[2] = 0.0f; 

    // Up is up (+Y)
    vUp[0] = 0.0f; vUp[1] = 1.0f; vUp[2] = 0.0f;

    // Forward is -Z (default OpenGL)
    vForward[0] = 0.0f; vForward[1] = 0.0f; vForward[2] = -1.0f;

    float fDelta = -15.0f;
    vOrigin[0] += vForward[0] * fDelta;
    vOrigin[1] += vForward[1] * fDelta;
    vOrigin[2] += vForward[2] * fDelta;

    float x[3], z[3];

    // Make rotation matrix
    // Z vector is reversed
    z[0] = -vForward[0];
    z[1] = -vForward[1];
    z[2] = -vForward[2];

    // X vector = Y cross Z 
    m3dCrossProduct3(x, vUp, z);

    // Matrix has no translation information and is
    // transposed.... (rows instead of columns)
    #define M(row,col)  m[col*4+row]
        M(0, 0) = x[0];
        M(0, 1) = x[1];
        M(0, 2) = x[2];
        M(0, 3) = 0.0;
        M(1, 0) = vUp[0];
        M(1, 1) = vUp[1];
        M(1, 2) = vUp[2];
        M(1, 3) = 0.0;
        M(2, 0) = z[0];
        M(2, 1) = z[1];
        M(2, 2) = z[2];
        M(2, 3) = 0.0;
        M(3, 0) = 0.0;
        M(3, 1) = 0.0;
        M(3, 2) = 0.0;
        M(3, 3) = 1.0;
    #undef M

    // Apply translation too
    float trans[16], M[16];
    m3dTranslationMatrix44(trans, -vOrigin[0], -vOrigin[1], -vOrigin[2]);  
    m3dMatrixMultiply44(M, m, trans);
    // Copy result back into m
    memcpy(m, M, sizeof(float)*16);
}

// 计算物体的坐标系
void GetObjectFrame(float* pfObjectFrame)
{
    // 物体的位置进行调节
    float vOrigin[3], vUp[3], vForward[3];

    // At origin
    vOrigin[0] = 0.0f; vOrigin[1] = 0.0f; vOrigin[2] = 0.0f; 

    // Up is up (+Y)
    vUp[0] = 0.0f; vUp[1] = 1.0f; vUp[2] = 0.0f;

    // Forward is -Z (default OpenGL)
    vForward[0] = 0.0f; vForward[1] = 0.0f; vForward[2] = -1.0f;

    // Calculate the right side (x) vector, drop it right into the matrix
    float vXAxis[3];
    m3dCrossProduct3(vXAxis, vUp, vForward);

    // Set matrix column does not fill in the fourth value...
    m3dSetMatrixColumn44(pfObjectFrame, vXAxis, 0);
    pfObjectFrame[3] = 0.0f;
    
    // Y Column
    m3dSetMatrixColumn44(pfObjectFrame, vUp, 1);
    pfObjectFrame[7] = 0.0f;       
                            
    // Z Column
    m3dSetMatrixColumn44(pfObjectFrame, vForward, 2);
    pfObjectFrame[11] = 0.0f;

   
    m3dSetMatrixColumn44(pfObjectFrame, vOrigin, 3);

    pfObjectFrame[15] = 1.0f;
}

void m3dCrossProduct3(float* result, const float* u, const float* v)
{
	result[0] =  u[1]*v[2] - v[1]*u[2];
	result[1] = -u[0]*v[2] + v[0]*u[2];
	result[2] =  u[0]*v[1] - v[0]*u[1];
}

void m3dTranslationMatrix44(float* m, float x, float y, float z)
{ m3dLoadIdentity44(m); m[12] = x; m[13] = y; m[14] = z; }

void m3dMatrixMultiply44(float* product, const float* a, const float* b )
{
#define A(row,col)  a[(col<<2)+row]
#define B(row,col)  b[(col<<2)+row]
#define P(row,col)  product[(col<<2)+row]

	// for (int i = 0; i < 3; i++) {
	// 	double ai0=A33(i,0),  ai1=A33(i,1),  ai2=A33(i,2);
	// 	P33(i,0) = ai0 * B33(0,0) + ai1 * B33(1,0) + ai2 * B33(2,0);
	// 	P33(i,1) = ai0 * B33(0,1) + ai1 * B33(1,1) + ai2 * B33(2,1);
	// 	P33(i,2) = ai0 * B33(0,2) + ai1 * B33(1,2) + ai2 * B33(2,2);
	// }

    for (int i = 0; i < 4; i++) {
		float ai0=A(i,0),  ai1=A(i,1),  ai2=A(i,2),  ai3=A(i,3);
		P(i,0) = ai0 * B(0,0) + ai1 * B(1,0) + ai2 * B(2,0) + ai3 * B(3,0);
		P(i,1) = ai0 * B(0,1) + ai1 * B(1,1) + ai2 * B(2,1) + ai3 * B(3,1);
		P(i,2) = ai0 * B(0,2) + ai1 * B(1,2) + ai2 * B(2,2) + ai3 * B(3,2);
		P(i,3) = ai0 * B(0,3) + ai1 * B(1,3) + ai2 * B(2,3) + ai3 * B(3,3);
	}


#undef A
#undef B
#undef P
}

// DEBUG
void DisplayMatrix(float* mat)
{
    using namespace std;
    cout << "=================" << endl;
    cout << mat[0] << " " <<  mat[1] << " " <<  mat[2] << " " <<  mat[3] << endl;
    cout << mat[4] << " " <<  mat[5] << " " <<  mat[6] << " " <<  mat[7] << endl;
    cout << mat[8] << " " <<  mat[9] << " " <<  mat[10] << " " <<  mat[11] << endl;
    cout << mat[12] << " " <<  mat[13] << " " <<  mat[14] << " " <<  mat[15] << endl;
    cout << "=================" << endl;
}

void m3dSetMatrixColumn44(float* dst, const float* src, const int column)
	{ memcpy(dst + (4 * column), src, sizeof(float) * 4); }