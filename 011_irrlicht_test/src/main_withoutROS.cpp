#include <iostream>
#include <irrlicht/irrlicht.h>
#include <GL/gl.h>
#include <GL/glext.h>

using namespace std;

#define WINDOW_WIDTH    640
#define WINDOW_HEIGHT   480


int main(int argc,char* argv[])
{
    cout<<"Irrlicht Test."<<endl;

    irr::IrrlichtDevice *device;
    device=irr::createDevice(
        irr::video::EDT_OPENGL,     //? 渲染器
        irr::core::dimension2d<irr::u32>(WINDOW_WIDTH,WINDOW_HEIGHT),   //窗口大小
        32,         //颜色深度
        false,      //是否全屏
        false,
        false);

    cout<<"1"<<endl;
    
    
    device->setWindowCaption(L"Irrlicht Test.");
    cout<<"2"<<endl;

    irr::video::IVideoDriver* driver;
    driver=device->getVideoDriver();    
    cout<<"3"<<endl;
    
    irr::scene::ISceneManager * scene;
    scene=device->getSceneManager();
    cout<<"4"<<endl;

    irr::scene::ISceneManager * view_template_scene;
    view_template_scene=scene->createNewSceneManager(false);
    cout<<"5"<<endl;

    if(device->run())
    {
        cout<<"Exit!"<<endl;
        //return 0;
    }

    //填充黑色背景
    driver->beginScene(true,true,irr::video::SColor(128,0,123,0));
    cout<<"6"<<endl;

    view_template_scene->drawAll();
    cout<<"7"<<endl;
    
    driver->endScene();
    cout<<"8"<<endl;

    cout<<"Done."<<endl;
    char a;
    cin>>a;


    return 0;
}