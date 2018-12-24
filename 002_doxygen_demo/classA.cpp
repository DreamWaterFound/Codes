#include "classA.h"

A::A()
{
    cout<<"类A的无参数构造函数被调用。"<<endl;
    mnNum=0;
}

A::A(int n)
{
    cout<<"类A的有参数构造函数被调用，参数值为"<<n<<endl;
    mnNum=n;
}

A::~A()
{
    cout<<"类A的析构函数被调用。"<<endl;
}

void A::setNum(int n)
{
    mnNum=n;
}

int A::getNum(void)
{
    return mnNum;
}
