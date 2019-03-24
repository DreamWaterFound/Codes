#ifndef __TYPE_SAMPLE_HPP
#define __TYPE_SAMPLE_HPP

typedef struct _SampleType
{
    long x;
    long y;
    int cluster_id;

    _SampleType(long _x,long _y):x(_x),y(_y){;}

}SampleType;






#endif //__TYPE_SAMPLE_HPP