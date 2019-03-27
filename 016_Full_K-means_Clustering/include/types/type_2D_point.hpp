#ifndef __TYPE_2D_point_HPP__
#define __TYPE_2D_point_HPP__

typedef struct _Points_2D
{
    long x;
    long y;
    //int cluster_id;   //这个被禁用了，具体的实现将会在K-means的类中进行

    _Points_2D(long _x,long _y):x(_x),y(_y){;}
    _Points_2D(void):x(0),y(0){;}


}Points_2D;

#endif //__TYPE_SAMPLE_HPP