# test_image_transfrom

在 test pytorch env 的基础上修改的， 所以有些注意事项也要看那个文件。

这个文件的目的是在之前的基础上，测试C++和Python之间的图像传输。

现在的思路是这样的，先使用C++结合OpenCV读取图像，然后将图像传递给Python程序，最后再由Python程序进行显示。

但是在进行所有的工作之前，我首先需要在之前的程序上进行一些调整。

会有这样的一个提示:

```
In file included from /usr/include/numpy/ndarraytypes.h:1777:0,
                 from /usr/include/numpy/ndarrayobject.h:18,
                 from /usr/include/numpy/arrayobject.h:4,
                 from /home/guoqing/Codes/025_CPP_call_Python/test_image_transform/src/main.cpp:18:
/usr/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
 #warning "Using deprecated NumPy API, disable it by " \
  ^
```

目前来看不影响最终的程序工作;但是尽管提示了我可以通过定义这个宏来解决,但是亲自测试,在每次编译的时候还是都会出现这个警告

