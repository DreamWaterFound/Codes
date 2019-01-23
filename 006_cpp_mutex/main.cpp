#include <iostream>
#include <thread>
#include <mutex>
using namespace std;

volatile int counter(0);
std::mutex mtx;

/**
 * @brief 线程的主函数
 * 
 */
void run(void);

/**
 * @brief 主函数
 * 
 * @param[in] argc 
 * @param[in] argv 
 * @return int 
 */
int main(int argc,char * argv[])
{
    cout<<"\e[1;35m - Main thread running.  \e[0m"<<endl;
    std::thread threads[10];
    //创建多个线程
    for(int i=0;i<10;++i)
    {
        threads[i] = std::thread(run);

    }
    //等待每个线程结束
    for(auto& th:threads)
        th.join();

    cout<<"cnt = "<<counter<<". "<<endl;

    return 0;
}


//定义这个宏来切换使用mtx.lock()和mtx.try_lock()
#define USE_TRY_LOCK

//多个进程的主函数
void run(void)
{
    for(int i=0;i<10000;i++)
    {
        #ifndef USE_TRY_LOCK
        mtx.lock();
        ++counter;
        mtx.unlock();
        #else
        if(mtx.try_lock())
        {
            ++counter;
            mtx.unlock();
        }
        #endif
    }
}