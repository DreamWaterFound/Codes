# 033 cpptoml test

对 TOML 配置文件的的解析测试，使用 [cpptoml](https://github.com/skystrife/cpptoml) 库，版本0.1.1.

**注意** TOML库本身也只是由头文件组成的，所以只需要在工程中放入该头文件即可。虽然它貌似也提供了cmake的方式，但是感觉和 cppopts 是类似的德性，所以还是直接使用头文件吧，简单又快捷。

里面的程序基本上都是原仓库中的 examples ，然后在其基础上添加了一些我的注释。

cmake要求3.1.0版本以上。