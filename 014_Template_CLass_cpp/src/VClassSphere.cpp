#ifndef __VCLASS_SPHER_CPP__
#define __VCLASS_SPHER_CPP__

#include "VClassSphere.hpp"

VolumeSphere::VolumeSphere(double r)
    :VolumeBase()
{
    _r=r;
}

double VolumeSphere::getVolume(void)
{
    return PI*_r*_r;
}

#endif //__VCLASS_SPHER_CPP__