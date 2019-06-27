# test pytorch env

这个的目的是为了测试C++调用Python库的时候，虚拟环境是否还能够work

已知问题和注意事项：
1、CMakeLists.txt中需要指向到自己电脑所使用的虚拟环境下面的库的目录
2、gcc/g++需要使用4.8版本
3、第一次cmake make 会出现nullptr无法识别的情况，然后重新进行cmake make 就可以解决了
4、ros添加的包的路径对python部分还是有影响，所以还是要考虑去除

**补充**
其实也可以不使用这个GCC、G++4.8版本来编译，前提是需要自己使用自己的编译器的版本来重新编译Python并且将其安装到对应的anaconda虚拟环境中。
