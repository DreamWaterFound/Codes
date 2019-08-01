#include <iostream>

using namespace std;

// 这个样子就可以了
void foo(void) __attribute__ ((deprecated));

int main()
{
    cout<<"Deprecated function test."<<endl;
    cout<<"Compled at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    foo();

    return 0;
}

void foo(void)
{
    cout<<"foo."<<endl;
}
