#ifndef __VCLASS_BASE_HPP__
#define __VCLASS_BASE_HPP__

class VolumeBase
{
public:
    VolumeBase();

    virtual double getVolume(void)=0;

public:
    double PI;

};

//#include "VClassVolumeBase.cpp"

#endif //__VCLASS_BASE_HPP__