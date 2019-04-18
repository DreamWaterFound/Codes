#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    cout<<"Bilateral Filter Demo."<<endl;

    if(argc!=2)
    {
        cout<<"Usage: "<<argv[0]<<" image_path"<<endl;
        return 0;
    }

    Mat src=imread(argv[1]);
    
    if(src.empty())
    {
        cout<<"Image "<<argv[1]<<" is empty! Please check it."<<endl;
        return 0;
    }

    Mat guassian,bilateral;

    GaussianBlur(src,guassian,Size(5,5),3,3);
    bilateralFilter(src,bilateral,-1,10,10);

    imshow("Origin",src);
    imshow("GaussianBlur",guassian);
    imshow("bilateralFilter",bilateral);

    waitKey(0);

    return 0;
}