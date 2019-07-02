# 032 cxxopts test

对轻量级工具 [cxxopts](https://github.com/jarro2783/cxxopts) 的测试。

这里是使用cmake的方式来find_package了，但是我死活都没有找到包含了头文件路径的变量。 -- 后来发现得使用别的方式

**注意** 还是不要作死去尝试在CMakeLists.txt中使用find_package命令找 cxxopts。这个作者写的cmake模块,Merge #53，实在是太非主流了，本来在cmake中include 一个dir就可以了非得link到，简单的事情搞的及其复杂，真的是想吐槽

**建议** 直接将其头文件包含到工程里面中---因为cxxopts这个库本身就是只有一个头文件组成；KinectFusionApp就是这样子做的。