#include <iostream>
#include <vector>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define PYRAMID_LAYER 8

int main(int argc,char *argv[])
{
    if(argc!=2)
    {
        cout<<"Usage:"<<endl;
        cout<<argv[0]<<" img_path"<<endl;
        return 0;
    }

    vector<Mat> imgs;
    imgs.resize(PYRAMID_LAYER);

    
    imgs[0]=imread(argv[1]);
    
    if(imgs[0].empty())
    {
        cout<<"Empty Image!"<<endl;
        return 0;
    }

    imshow("Origin Picture (layer #0)",imgs[0]);

    for(int i=1;i<PYRAMID_LAYER;i++)
    {
        
        pyrDown(imgs[i-1],imgs[i]);
        ostringstream ss;
        ss<<"Layer #"<<i;
        imshow(ss.str(),imgs[i]);
    }

    waitKey(0);
    destroyAllWindows();

    imshow("Pyramid way", imgs[1]);
    //下面是通过金字塔方式和resize方式所进行的对比
    Mat pic;
    resize(imgs[0],pic,Size(0,0),0.5,0.5);
    imshow("resize way",pic);
    
    waitKey(0);
    destroyAllWindows();




    return 0;
}