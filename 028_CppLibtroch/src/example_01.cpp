#include <iostream>

#include <torch/torch.h>

using namespace std;

int main()
{
    cout<<"C++ depoly torch model test."<<endl;
    cout<<"No params needed."<<endl;
    cout<<"Complied at "<<__TIME__<<" , "<<__DATE__<<"."<<endl;

    torch::Tensor tensor=torch::rand({2,3});
    cout<<"tensor="<<tensor<<endl;

    return 0;
}