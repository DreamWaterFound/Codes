#include <iostream>
#include <string>
#include <sstream>

#include "tools/SampleSets_2D_point.hpp"

using namespace std;

int main(int argc, char* argv[])
{
    cout<<"K-means Test - Create 2D Sample Tool."<<endl<<endl;

    if(argc<3)
    {
        cout<<"Usage: "<<argv[0]<<" range_height range_width file_path"<<endl;
        cout<<"Example: "<<argv[0]<<" 240 320 ./data/test.txt"<<endl;
        return 0;
    }

    size_t height,width;
    stringstream ss(argv[1]);
    ss>>height;
    ss=stringstream(argv[2]);
    ss>>width;

    Samples sample;

    cout<<"Click the image to set sample points. When you are ready, press ESC or just close the window. Waiting ... ";
    sample.getSamples(height,width);
    cout<<"Complete."<<endl;

    cout<<"Saving to "<<argv[3]<<" ... ";
    if(sample.saveSamples(string(argv[3])))
    {
        cout<<"Complete."<<endl;
    }
    else
    {
        cout<<"Failed."<<endl;
    }
   
    return 0;
}