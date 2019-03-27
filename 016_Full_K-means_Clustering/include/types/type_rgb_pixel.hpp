#ifndef __TYPE_RGB_PIXEL_HPP__
#define __TYPE_RGB_PIXEL_HPP__

typedef struct _RGB_PIXEL
{
    //主要是考虑到会求和,所以还是使用了long
    long r;
    long g;
    long b;

    long u;
    long v;
    

    _RGB_PIXEL(long _u,long _v,long _r,long _g,long _b):u(_u),v(_v),r(_r),g(_g),b(_b){;}
    _RGB_PIXEL(void):u(0),v(0),r(0),g(0),b(0){;}


}RGB_PIXEL;

#endif //__TYPE_RGB_PIXEL_HPP__