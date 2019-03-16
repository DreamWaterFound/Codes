#include <iostream>

#include "Tclass.hpp"
#include "VClassSphere.hpp"

using namespace std;

int main(int argc,char* argv[])
{
    //对于模板类的测试
    Tclass<double> tc(3.14,2.36);
    cout<<"sum is "<<tc.add()<<endl;

    VolumeSphere vv(6);
    cout<<"vol of 6 is "<<vv.getVolume()<<endl;

    return 0;
}