/**
 * @file kernel.h
 * @author guoqing (1337841346@qq.com)
 * @brief 头文件
 * @version 0.1
 * @date 2019-11-27
 * 
 * @copyright Copyright (c) 2019
 * 
 */


#define DIM     600                     // 图像长宽  
#define PI      3.1415926535897932f     // 圆周率
  

// globals needed by the update routine  
// 其实里面就是只有一个图像指针
struct DataBlock  
{  
    unsigned char   *dev_bitmap;  
};  

// Launcher
void KernelLauncher(unsigned char ooo);
