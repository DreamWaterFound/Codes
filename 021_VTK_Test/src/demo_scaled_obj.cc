#include "vtkIncludes.hpp"



class vtkMyCallback : public vtkCommand
{
public:
  static vtkMyCallback *New()             //vkt的实例化都是通过New()来实现的</span>
    { return new vtkMyCallback; }
  virtual void Execute(vtkObject *caller, unsigned long, void*)
    {
      vtkTransform *t = vtkTransform::New();
      vtkBoxWidget *widget = reinterpret_cast<vtkBoxWidget*>(caller);
      widget->GetTransform(t);
      widget->GetProp3D()->SetUserTransform(t);
      t->Delete();
    }
};
 

int main()
{
    //数据源对象, 建立一个圆锥
    vtkConeSource *cone = vtkConeSource::New();
    //设定各种参数
    cone->SetHeight( 3.0 );
    cone->SetRadius( 1.0 );
    cone->SetResolution( 100 );

    //映射器，接受cone的输出，将数据映射为几何元素
    vtkPolyDataMapper *coneMapper = vtkPolyDataMapper::New();
    coneMapper->SetInputConnection( cone->GetOutputPort() ); 

    //演员出场
    vtkActor *coneActor = vtkActor::New();
    coneActor->SetMapper( coneMapper );

    //绘制器
    vtkRenderer *ren1= vtkRenderer::New();
    ren1->AddActor( coneActor );
    ren1->SetBackground( 0.1, 0.2, 0.4 );

    //通过AddRenderer将renderer绘制到窗口上
    vtkRenderWindow *renWin = vtkRenderWindow::New();
    renWin->AddRenderer( ren1 );
    renWin->SetSize( 640, 480 );

    //vtkRenderWindowInteracor类捕捉窗口的鼠标的键盘事件，然后分配给其他的类。
    vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
    iren->SetRenderWindow(renWin);

    //要在vtk中添加新的交互方式需要从tkInteractorStyle 类中派生新的类，如 vtkInteractorStyleTrackballCamera实现操作杆交互方式，对相机进行交互。
    vtkInteractorStyleTrackballCamera *style = vtkInteractorStyleTrackballCamera::New(); 
    iren->SetInteractorStyle(style);//使用vtkBoxWidget对象来对actor进行相应的转换变形，生成窗口小部件，SetInteractor方法建立交互 
    vtkBoxWidget *boxWidget = vtkBoxWidget::New();
    boxWidget->SetInteractor(iren); 
    boxWidget->SetPlaceFactor(1.0);  //缩放系数 
    boxWidget->SetProp3D(coneActor);
    boxWidget->PlaceWidget(); 
    vtkMyCallback *callback = vtkMyCallback::New();
    //每个vtk类都提供一个AddObserver方法建立回调，如果Observer监控的对象的一个事件被触发，则一个响应的回调函数就会被调用。在boxWidget上为InterractionEvent建  //立一个Observer,只要运行Excute(），就进行回调
    boxWidget->AddObserver(vtkCommand::InteractionEvent, callback);


    boxWidget->On();


    iren->Initialize();
    iren->Start();


    cone->Delete();
    coneMapper->Delete();
    coneActor->Delete();
    callback->Delete();
    boxWidget->Delete();
    ren1->Delete();
    renWin->Delete();
    iren->Delete();
    style->Delete();

    return 0;
}
