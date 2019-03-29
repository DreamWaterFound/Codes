#ifndef __TYPE_RGBD_PIXEL_HPP__
#define __TYPE_RGBD_PIXEL_HPP__

typedef struct _RGBD_PIXEL
{
    //主要是考虑到会求和,所以还是使用了long
    long r;
    long g;
    long b;
    long long d;

    long u;
    long v;
    

    _RGBD_PIXEL(long _u,long _v,long _r,long _g,long _b,long long _d):u(_u),v(_v),r(_r),g(_g),b(_b),d(_d){;}
    _RGBD_PIXEL(void):u(0),v(0),r(0),g(0),b(0),d(0.0){;}


}RGBD_PIXEL;

#endif //__TYPE_RGBD_PIXEL_HPP__