#include <iostream>
#include "classA.h"
#include "classB.h"

using namespace std;

int main(int argc,char *argv[])
{
    cout<<"Doxygen Demo."<<endl;

    A obj(2);
    B objb(1);

    cout<<"准备析构。"<<endl;

    return 0;
}