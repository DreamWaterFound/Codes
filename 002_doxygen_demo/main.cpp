#include <iostream>
#include "classA.h"
#include "classB.h"

using namespace std;

//这个文件是主文件。

int main(int argc,char *argv[])
{
    cout<<"Doxygen Demo."<<endl;

    A obj(2);
    B objb(1);

    cout<<"准备析构。"<<endl;

    return 0;
}