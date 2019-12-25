
#include <opencv2/opencv.hpp>
#include "kernel.h"
#include "iostream"

using std::cout;
using std::cin;
  
int main(void)  
{  
    while(1)
    {
        for(size_t i=15;i>=9;--i)
        {
            KernelLauncher((unsigned char)i);
        }
    }

    return 0;
}
