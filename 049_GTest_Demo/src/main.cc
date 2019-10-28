#include <iostream>
#include <Add.h>

using std::cin;
using std::cout;
using std::endl;


int main(int argc, char* argv[])
{
    cout<<"Test Google Test."<<endl;
    cout<<"Complied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    int a,b;
    cout<<endl;
    cout<<"Input Num a:";
    cin>>a;
    cout<<"Input Num b:";
    cin>>b;
    cout<<"Res = "<<Add(a,b)<<endl;

    return 0;
}