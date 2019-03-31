# 018_SSE_AVX_Accelerate

使用Intel SSE 和 AVX 指令集对程序进行加速的尝试

PS：其实现在C++的编译器所进行的优化已经非常好了，单独在C++程序中使用SSE和AVX指令集加速已经没有明显的效果，但是我还是想试一试。

# Tools

为了避免出现分支预测，需要一系列的无规律的浮点数作为测试数据；而不同方式下的对比则需要使用相同的测试数据来避免偶然性。因此这里提供了工具 generateData 来产生指定数目的随机浮点数并且存储到外部文件中以供测试程序加载。

# Notice

使用SSE加速的程序源码，在表现形式上，windows下和linux下是不太一样的！包含的文件不一样，写法也不太一样。例如对齐操作：

```
//windows 下的写法
__declspec(align(16)) float op2[4] = {1.0, 2.0, 3.0, 4.0};  
//linux 下的的写法
float op2[4] __attribute__((aligned(16))) = {1.0, 2.0, 3.0, 4.0};  
```

类似地在头文件上也有不相同的地方：

```
//windows
#include <intrin.h> 
//linux
#include "emmintrin.h"
#include "tmmintrin.h"
```

目前这里的程序都是在linux上进行的，因此源码中的有关写法也都是按照linux下g++的要求来写的。



