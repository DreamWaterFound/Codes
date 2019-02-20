//尝试在 05_SimpleDisplay 的基础上禁用全屏快捷键

#include <iostream>
#include <pangolin/pangolin.h>

//随意定义了一个结构体数据类型,其中包含一个整数,一个浮点数和一个字符串
struct CustomType
{
  CustomType()
    : x(0), y(0.0f) {}

  CustomType(int x, float y, std::string z)
    : x(x), y(y), z(z) {}

  int x;
  float y;
  std::string z;
};
//定义了上面这个结构体的流输入输出符
std::ostream& operator<< (std::ostream& os, const CustomType& o){
  os << o.x << " " << o.y << " " << o.z;
  return os;
}

std::istream& operator>> (std::istream& is, CustomType& o){
  is >> o.x;
  is >> o.y;
  is >> o.z;
  return is;
}

void SampleMethod()
{
    std::cout << "You typed ctrl-r or pushed reset" << std::endl;
}

void NonFullScreen(void)
{
  return ;
}


int main(/*int argc, char* argv[]*/)
{  
  // Load configuration data
  //pangolin::ParseVarsFile("app.cfg");

  // Create OpenGL window in single line
  pangolin::CreateWindowAndBind("Main",640,480);
  
  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
    pangolin::ModelViewLookAt(-0,0.5,-3, 0,0,0, pangolin::AxisY)
  );

  //侧边栏的宽度
  const int UI_WIDTH = 180;

  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

  // Add named Panel and bind to variables beginning 'ui'
  // A Panel is just a View with a default layout and input handling
  pangolin::CreatePanel("ui")
      .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

  // Safe and efficient binding of named variables.
  // Specialisations mean no conversions take place for exact types
  // and conversions between scalar types are cheap.
  pangolin::Var<bool> a_button(   //注意按钮必须是bool型
    "ui.A_Button",                //句柄和名称
    false,                        //初始状态是否是选中状态
    false);                       //是否是复选框
  pangolin::Var<double> a_double(
    "ui.A_Double",                //句柄和名称
    2,                            //开始运行时的初始值
    -10,                          //滑动条下限
    10);                          //滑动条上限
  pangolin::Var<int> an_int("ui.An_Int",2,4,5);   //同上
  pangolin::Var<double> a_double_log(
    "ui.Log_scale var",           //句柄和名称
    500,                            //开始运行时的初始值
    1E2,                            //上限
    1E4,                          //下限
    false);                        //是否使用对数表示
  pangolin::Var<bool> a_checkbox("ui.A_Checkbox",false,true);   //同上
  pangolin::Var<int> an_int_no_input("ui.An_Int_No_Input",2);   //如果什么都不加的话，那么就无法通过鼠标来控制它
  pangolin::Var<CustomType> any_type("ui.Some_Type", CustomType(0,1.2f,"Hello") );  //显示其他的数据类型，但是要求这个数据类型要能够像上面一样定义流输入和流输出函数

  //定义具有两个特殊功能的按钮
  pangolin::Var<bool> save_window("ui.Save_Window",false,false);
  pangolin::Var<bool> save_cube("ui.Save_Cube",false,false);
  pangolin::Var<bool> record_cube("ui.Record_Cube",false,false);

  // std::function objects can be used for Var's too. These work great with C++11 closures.
  //详细资料可以看这里: https://blog.csdn.net/janeqi1987/article/details/87181517
  //支持这个其实真的是一个不错的特性
  pangolin::Var<std::function<void(void)> > reset("ui.Reset", SampleMethod);    //自定义按钮的响应函数

  // Demonstration of how we can register a keyboard hook to alter a Var
  //这里的功能就是按一下按键就会修改成为指定的数值
  //快捷键的响应,以及相关显示数值的修改
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'b', pangolin::SetVarFunctor<double>("ui.A_Double", 3.5));
  //快捷键的响应,并且可以触发一个自定义的函数
  // Demonstration of how we can register a keyboard hook to trigger a method
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'r', SampleMethod);

  pangolin::RegisterKeyPressCallback('\t', NonFullScreen);

  // Default hooks for exiting (Esc) and fullscreen (tab).
  //NOTICE 这里默认是绑定了上面的两个按键
  while( !pangolin::ShouldQuit() )
  {
    // Clear entire screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    

    //判断一个按钮对象是否按下
    //NOTE 如果这个是复选框按钮的话,执行下面的语句将会使得这个按钮无法显示出被按下的情况
    
    if( pangolin::Pushed(a_button) )
      std::cout << "You Pushed a button!" << std::endl;
      
    // Overloading of Var<T> operators allows us to treat them like
    // their wrapped types, eg:
    if( a_checkbox )
      an_int = (int)a_double;

    if( !any_type->z.compare("robot"))
        any_type = CustomType(1,2.3f,"Boogie");

    an_int_no_input = an_int;

    if( pangolin::Pushed(save_window) )
        pangolin::SaveWindowOnRender("window");

    if( pangolin::Pushed(save_cube) )
        d_cam.SaveOnRender("cube");
    
    if( pangolin::Pushed(record_cube) )
        pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=50,bps=8388608,unique_filename]//screencap.avi");

    // Activate efficiently by object
    //TODO 这个我可以理解为,当绘制的物体发生变动时,窗口也将会被处于激活状态?
    d_cam.Activate(s_cam);

    // Render some stuff
    //TODO 目前还不知道这个是做什么的
    glColor3f(1.0,1.0,1.0);
    pangolin::glDrawColouredCube();

    // Swap frames and Process Events
    pangolin::FinishFrame();
  }

  return 0;
}
