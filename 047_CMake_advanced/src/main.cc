#include <iostream>

#include "dummy.hpp"
#include "Config.h"

using std::cout;
using std::endl;


int main(int argc, char* argv[])
{
    cout<<"Ok!"<<endl;
    cout<<"Complied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    #ifdef MY_FLAG
        cout<<"MY_FLAG defined."<<endl;
    #else
        cout<<"MY_FLAG undefined."<<endl;
    #endif

    cout<<"DATAAAAAAA = "<<DATAAAAAAA<<endl;

    my_print();

    return 0;
}
