#include "classB.h"

//类B的实现文件

B::B()
{
    cout<<"类B的无参数构造函数被调用。"<<endl;
    mnNum=0;
}

B::B(int n)
{
    cout<<"类B的有参数构造函数被调用，参数值为"<<n<<endl;
    mnNum=n;
}

B::~B()
{
    cout<<"类B的析构函数被调用。"<<endl;
}

void B::setNum(int n)
{
    mnNum=n;
}

int B::getNum(void)
{
    return mnNum;
}
