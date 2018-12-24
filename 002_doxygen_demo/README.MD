# 使用doxygen产生文本文档的实验

----

# 一、功能简述

文件中自定义了三个没有什么实际功能的文件：

- main.cpp
- classA.h  & classA.cpp
- classB.h & classB.cpp

注意，**不要尝试去编译它**，肯定会有错的；但是这个并不会影响这个实验的结果。

Doxygen的配置文件在./doc/Doxygen，因此当在工程路径下需要这样执行：

```
$ doxygen ./doc/Doxygen
```

生成的文档将会存储在./doc/html下，打开index.html即可在浏览器中查看。

这里在配置文件中没有生成latex文件。

# 二、编译

## 2.1 依赖

### 必要依赖

- doxygen (实验中使用的是1.8.11版本)

### 可选依赖

无

## 2.2 开发平台和工具

- Ubuntu 16.04
- VScode

## 2.3 编译工具链

- cmake 3.5.1
- g++ 5.4.0

## 2.4 其他

无

# 三、注意事项

无

# 四、参考

- [Doxygen 10 分钟入门教程](http://cedar-renjun.github.io/2014/03/21/learn-doxygen-in-10-minutes/index.html)
- [DOXYGEN简明实用教程](https://my.oschina.net/u/2306127/blog/484501)
- [Doxygen注释规范](https://blog.csdn.net/qq_25779555/article/details/77737391)



