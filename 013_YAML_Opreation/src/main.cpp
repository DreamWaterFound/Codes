#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    if(argc!=2)
    {
        cout<<"Usage: "<<argv[0]<<" <yaml file path>"<<endl;
        return 0;
    }

    cout<<"Openning yaml file "<<argv[1]<<" ..."<<endl;
    cv::FileStorage fs(argv[1],cv::FileStorage::READ);

    if(!fs.isOpened())
    {
        cout<<"Failed."<<endl;
        return 0;
    }
    
    cout<<"Opened."<<endl;

    int num=(int)fs["test_num_a"];
    cout<<"test_num_a = "<<num<<endl;

    double test_float=(double)fs["test_float"];
    cout<<"test_float = "<<test_float<<endl;

    //直接进行转换貌似有问题，所以可以通过这种方式进行转换
    bool test_bool = (bool)(int)(fs["test_bool"]);
    cout<<"test_bool = "<<test_bool<<"    ";
    if(test_bool) cout<<"True"<<endl;
    else  cout<<"Flase"<<endl;

    string str=(string)fs["test_str"];
    cout<<"test_str="<<str<<endl;

    cv::FileNode vec=fs["test_vec"];
    if(vec.size()!=3)
    {
        cout<<"Wrong vec size!"<<endl;
    }

    cout<<"test vec = ["<<(int)vec[0]<<" "<<(int)vec[1]<<" "<<(int)vec[2]<<"]"<<endl;


    


    fs.release();
    return 0;
}