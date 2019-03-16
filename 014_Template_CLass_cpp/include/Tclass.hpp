//模板类的声明
#ifndef __TCLASS_HPP__
#define __TCLASS_HPP__

template <class T>
class Tclass
{
public:
    Tclass(T a,T b);

    T add(void);
private:
    T _a;
    T _b;

};

//避免链接不到的问题;
//TODO 但是问题是如果在Cmake中也包含了会出现重定义问题
#include "Tclass.cpp"

#endif //__TCLASS_HPP__