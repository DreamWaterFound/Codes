#include "vtkBMPReader.h"
#include "vtkImageViewer.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkImageMagnify.h"
#include "vtkTransform.h"
  
#include <iostream>
using namespace std;

int main(int argc,char *argv[])
{
    if(argc!=2)
    {
        cout<<"Usage: "<<argv[0]<<" BMP_image_path"<<endl;
        return 0;
    }

    vtkBMPReader  *reader=  vtkBMPReader::New();
    reader->SetDataByteOrderToLittleEndian();
    reader->SetFileName(argv[1]); //程序当前目录中有这个文件
    reader->SetDataOrigin(0,0,0.0);

    vtkTransform *t1=vtkTransform::New();  
    t1->RotateZ(0);
    reader->SetTransform(t1); //控制图像的旋转
    vtkImageMagnify *scale=vtkImageMagnify::New();
      scale->SetInputConnection(reader->GetOutputPort());
      scale->SetMagnificationFactors(2,1.5,1.5); //图像各个维度的维放

    vtkImageViewer  *viewer = vtkImageViewer::New();
    viewer->SetInputConnection(scale->GetOutputPort());
    viewer->SetColorWindow(1000);
    viewer->SetColorLevel(200);
    viewer->SetPosition(0,0);
    viewer->Render();

    vtkRenderWindowInteractor *viewerinter = vtkRenderWindowInteractor::New();
    viewer->SetupInteractor(viewerinter);

    vtkImageViewer  *viewer2 = vtkImageViewer::New(); //没有缩放的原图，以作对比
    viewer2->SetInputConnection(reader->GetOutputPort());
    viewer2->SetColorWindow(256);
    viewer2->SetColorLevel(200);
    viewer2->SetPosition(0,100);
    viewer2->Render();

    vtkRenderWindowInteractor *viewerinter2 = vtkRenderWindowInteractor::New();
    viewer2->SetupInteractor(viewerinter2);

    // 堵塞了
    // viewerinter->Initialize();
    // viewerinter->Start();  
    // viewerinter2->Initialize();
    // viewerinter2->Start();  
    char c;
    cin>>c;
   
    scale->Delete();
    viewer->Delete();
    viewerinter->Delete();
    viewer2->Delete();
    viewerinter2->Delete();
    return 0;
}