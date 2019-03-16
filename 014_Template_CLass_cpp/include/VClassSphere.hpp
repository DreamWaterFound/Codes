#ifndef __VCLASS_SPHER_HPP__
#define __VCLASS_SPHER_HPP__

#include "VclassBase.hpp"

class VolumeSphere : public VolumeBase
{
public:
    VolumeSphere(double);

    double getVolume(void);

protected:

    double _r;

};

#include "VClassVolumeBase.cpp"
#include "VClassSphere.cpp"


#endif //__VCLASS_SPHER_HPP__