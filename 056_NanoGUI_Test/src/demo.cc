/**
 * @file demo.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 尝试使用 Nano GUI 的例子
 * @version 0.1
 * @date 2020-01-20
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <nanogui/nanogui.h>
#include <iostream>
#include <Eigen/Core>

#include <thread>
#include <chrono>
#include <mutex>
#include <sstream>
#include <string>

// 决定是否将Viewer放在另外的一个线程中进行显示
#define USE_MULTI_THREAD true

// using namespace nanogui;
using std::cout;
using std::endl;



bool bThreadDone = false;
std::mutex mutexThreadState;


// 自定义
class AppScreen : public nanogui::Screen
{
public:
    // 应该在构造函数中完成窗口部件的布局设计
    AppScreen():
        nanogui::Screen(
            Eigen::Vector2i(1366, 768),
            "Guoqing NanoGUI Test",
            true)
    {
        // using namespace nanogui;

        nanogui::Window *pWindow = new nanogui::Window(this, "My demo");
        pWindow->setPosition(Eigen::Vector2i(100,100));
        pWindow->setLayout(new nanogui::GroupLayout);

        new nanogui::Label(pWindow, "Label", "sans-bold", 18);

        nanogui::Button *pButton = new nanogui::Button(pWindow, "Normal Buttion");
        pButton->setCallback([]{cout<<"Plain button: pushed!"<<endl;});
        pButton->setTooltip("A plain buttion.");

        new nanogui::Label(pWindow, "Progress Bar", "sans-bold", 18);
        nanogui::Widget *pWidget = new Widget(pWindow);
        pWidget->setLayout(
            new nanogui::BoxLayout(
                nanogui::Orientation::Horizontal,
                nanogui::Alignment::Middle,
                0,6
            )
        );

        mpProgress = new nanogui::ProgressBar(pWidget);

        new nanogui::Label(pWindow, "Progress Bar", "sans-bold", 18);
        
        mpTextBox = new nanogui::TextBox(pWidget);
        mpTextBox->setFixedSize(Eigen::Vector2i(60, 25));
        mpTextBox->setValue("0");
        mpTextBox->setUnits("%");
        
        mpTheme = new nanogui::Theme(*(mpTextBox->theme()));
        
        mpTheme->mTextColor=nanogui::Color(0,255,0,200);
        mpTheme->mTransparent = nanogui::Color(255,255,255,200);
        mpTextBox->setTheme(mpTheme);
        // mpTextBox->


    }

    ~AppScreen(){}

    // 这里定义按键的响应函数
    virtual bool keyboardEvent(int key, int scancode, int action, int modifiers) 
    {
        if (Screen::keyboardEvent(key, scancode, action, modifiers))
            return true;
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) 
        {
            setVisible(false);
            return true;
        }
        
        return false;
    }

    // 这里是对部件的绘制内容
    virtual void draw(NVGcontext *ctx) {

        float fProgressValue;
        
        /* Animate the scrollbar */
        if(!USE_MULTI_THREAD)
        {
            std::lock_guard<std::mutex> lock(mMutexProgress);
            fProgressValue = mfPBarValue = std::fmod((float) glfwGetTime() / 10, 1.0f);
        }
        else
        {
            std::lock_guard<std::mutex> lock(mMutexProgress);
            fProgressValue = mfPBarValue;
        }


        mpProgress->setValue(fProgressValue);


        std::stringstream ss;
        ss<<static_cast<int>(fProgressValue*100);

        mpTextBox->setValue(ss.str().c_str());

        /* Draw the user interface */
        Screen::draw(ctx);
    }

    // 在这里定义绘制的背景内容
    virtual void drawContents() {

    }

    // 供外部程序设置进度条进度的, 外部线程调用
    void setProgressBarValue(float fValue)
    {
        std::lock_guard<std::mutex> lock(mMutexProgress);
        mfPBarValue = fValue;
    }

private:

    nanogui::ProgressBar *mpProgress;
    nanogui::TextBox     *mpTextBox;
    nanogui::Theme       *mpTheme;

    std::mutex mMutexProgress;
    float mfPBarValue = 0.0f;

};

AppScreen *pScreen = nullptr;


void thread_run(void)
{
    {
        std::lock_guard<std::mutex> lock(mutexThreadState);
        bThreadDone = false;
    }

    nanogui::init();
    pScreen = new AppScreen();

    pScreen->setBackground(nanogui::Color(0.2f, 0.2f, 0.2f, 1.0f));
    pScreen->performLayout();
    pScreen->drawAll();
    pScreen->setVisible(true);

    nanogui::mainloop();
    nanogui::shutdown();

    {
        std::lock_guard<std::mutex> lock(mutexThreadState);
        bThreadDone = true;
    }

    cout<<"Viewer Exited!"<<endl;

}


int main(int argc, char* argv[])
{
    cout<<"NanoGUI Test."<<endl;
    cout<<"Complied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;


    if(USE_MULTI_THREAD)
    {
        cout<<"Multi thread mode."<<endl;
        
        std::thread* pThread = new std::thread(thread_run);

        bool bThreadState = false;
        float fProgressValue = 0.0f;

        // 确保子进程中的内容已经创建完成
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        while(!bThreadState)
        {
            // 更新进度条
            fProgressValue = fProgressValue > 1.0? 0.0 : fProgressValue + 0.01;
            
            pScreen->setProgressBarValue(fProgressValue);

            // 当前进程睡眠
            std::this_thread::sleep_for(std::chrono::milliseconds(30));

            // 更新进程状态
            {
                std::lock_guard<std::mutex> lock(mutexThreadState);
                bThreadState = bThreadDone;
            }
        };
        pThread->join();
    }
    else
    {
        cout<<"Single thread mode."<<endl;
        // 下面先按照在单线程中运行
        nanogui::init();
        pScreen = new AppScreen();

        pScreen->setBackground(nanogui::Color(0.2f, 0.2f, 0.2f, 1.0f));
        pScreen->performLayout();
        pScreen->drawAll();
        pScreen->setVisible(true);

        nanogui::mainloop();
        nanogui::shutdown();
    }
    

   


    return 0;
}
