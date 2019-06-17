# test_image_transfrom

在 test pytorch env 的基础上修改的， 所以有些注意事项也要看那个文件。

这个文件的目的是在之前的基础上，测试C++和Python之间的图像传输。

现在的思路是这样的，先使用C++结合OpenCV读取图像，然后将图像传递给Python程序，最后再由Python程序进行显示。

但是在进行所有的工作之前，我首先需要在之前的程序上进行一些调整。