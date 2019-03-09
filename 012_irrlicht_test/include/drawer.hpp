#include <irrlicht/irrlicht.h>
#include <GL/gl.h>
#include <GL/glext.h>

#define WINDOW_WIDTH    640
#define WINDOW_HEIGHT   480

class Drawer
{

public:
    
/**
 * @brief 构造函数
 * 
 */
Drawer()
{
    
    mpDevice=irr::createDevice(
        irr::video::EDT_OPENGL,     //? 渲染器
        irr::core::dimension2d<irr::u32>(WINDOW_WIDTH,WINDOW_HEIGHT),   //窗口大小
        32,         //颜色深度
        false,      //是否全屏
        false,
        false);

    
    mpDevice->setWindowCaption(L"Irrlicht with ROS test.");

    mpDriver=mpDevice->getVideoDriver(); 

    mpScene=mpDevice->getSceneManager();

    mpViewTemplateScene=mpScene->createNewSceneManager(false);

    mpDevice->run();
    

    
    //填充黑色背景
    mpDriver->beginScene(true,true,irr::video::SColor(128,0,123,0));

    mpViewTemplateScene->drawAll();

    mpDriver->endScene();
    

    mpDevice->drop();

}
~Drawer()
{
    //mpViewTemplateScene->drop();
    //    mpDevice->drop();
}

private:
    irr::IrrlichtDevice *mpDevice;
    irr::video::IVideoDriver* mpDriver;
    irr::scene::ISceneManager * mpScene;
    irr::scene::ISceneManager * mpViewTemplateScene;

};
