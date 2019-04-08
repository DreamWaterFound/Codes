#include <vtkImageResize.h>
#include "vtkImageSincInterpolator.h"

#include <vtkVersion.h>
#include <vtkImageActor.h>
#include <vtkImageCanvasSource2D.h>
#include <vtkImageData.h>
#include <vtkJPEGReader.h>
#include <vtkImageMapper3D.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkSmartPointer.h>

#include <iostream>
using namespace std;
 
int main(int argc, char *argv[])
{
  vtkSmartPointer<vtkImageData> imageData;

  int newSize[2] = {200, 10};
  int windowFunction = 0;

  // Verify input arguments
  if(argc<4 && argc>5)
  {
      cout<<"Uasge: "<<argv[0]<<" jpg_image_path resize_x resize_y "<<endl;
      return 0;
  }


    //Read the image
    //加载 jpg
    vtkSmartPointer<vtkJPEGReader> jpegReader =  vtkSmartPointer<vtkJPEGReader>::New();
    jpegReader->SetFileName ( argv[1] );
    jpegReader->Update();

    imageData = jpegReader->GetOutput();

    newSize[0] = atoi(argv[2]);
    newSize[1] = atoi(argv[3]);

    if (argc > 4)
    {
     windowFunction = atoi(argv[4]);
    }
   


  vtkSmartPointer<vtkImageSincInterpolator> interpolator =  vtkSmartPointer<vtkImageSincInterpolator>::New();
  if (windowFunction >= 0 && windowFunction <= 10)
  {
    interpolator->SetWindowFunction(windowFunction);
  }

  vtkSmartPointer<vtkImageResize> resize = vtkSmartPointer<vtkImageResize>::New();
#if VTK_MAJOR_VERSION <= 5
  resize->SetInput(imageData);
#else
  resize->SetInputData(imageData);
#endif
  resize->SetOutputDimensions(newSize[0], newSize[1], 1);
    // resize->SetOutputDimensions(640, 480, 1);
  resize->Update();

  if (windowFunction < 0)
  {
    resize->InterpolateOff();
    std::cout << "Using nearest neighbor interpolation" << std::endl;;
    }
  else
    {
    std::cout << "Using window function : "
              << interpolator->GetWindowFunctionAsString() << std::endl;;
    }

  // Create an image actor to display the image
  vtkSmartPointer<vtkImageActor> imageActor =
    vtkSmartPointer<vtkImageActor>::New();
  imageActor->GetMapper()->SetInputConnection(resize->GetOutputPort());

 // Setup renderer
  vtkSmartPointer<vtkRenderer> renderer =
    vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(imageActor);
  renderer->ResetCamera();

  // Setup render window
  vtkSmartPointer<vtkRenderWindow> renderWindow =
    vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  renderWindow->SetSize(640,480);
  renderWindow->SetWindowName("Pic");
  renderer->SetViewport(0,0,1,1);


  // Setup render window interactor
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  vtkSmartPointer<vtkInteractorStyleImage> style =
    vtkSmartPointer<vtkInteractorStyleImage>::New();

  renderWindowInteractor->SetInteractorStyle(style);

  // Render and start interaction
  renderWindowInteractor->SetRenderWindow(renderWindow);
  renderWindowInteractor->Initialize();

  renderWindowInteractor->Start();

  return EXIT_SUCCESS;
}