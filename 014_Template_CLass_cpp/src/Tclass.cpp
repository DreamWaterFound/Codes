#ifndef __TCLASS_CPP__
#define __TCLASS_CPP__

#include "Tclass.hpp"

template<class T>
Tclass<T>::Tclass(T a,T b)
{
    _a=a;_b=b;
}

template<class T>
T Tclass<T>::add(void)
{
    return _a+_b;
}

#endif //__TCLASS_CPP__