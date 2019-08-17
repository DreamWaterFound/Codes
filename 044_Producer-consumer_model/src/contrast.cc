#include<condition_variable>
#include<mutex>
#include<thread>
#include<iostream>
#include<queue>
#include<chrono>
#include <string>
#include <cmath>

using namespace std;



// 假想的结构体
struct ProductObject
{
    size_t      id;
    std::string content;

    ProductObject(size_t _id, std::string _strContent):id(_id),content(_strContent){}
    ProductObject(){}

    friend ostream& operator<<(ostream& os, const ProductObject& my)
    {
        os<<my.content;
        return os;
    }
};



// 这里所使用的信号量都是使用全局变量,实际应用中多是类的成员变量这个样子

mutex mutexAB;                      // 线程AB间交换的数据
ProductObject firstProduct;         // 线程A生产的产品

mutex mutexBA;                      // 线程BA之间交换的数据
ProductObject secondProduct;        // 线程B处理的产品

ProductObject finalProduct;         // 线程A进行最终处理之后得到的产品

mutex mutexKeepRun;                 // 下面对应的锁
bool keepRun;                       // 是否保持运行

// 避免死锁
mutex mutexThreadBState;
// bool bIsThreadBOK;
bool bIsBAOK;                       // 用于这两个交换数据的条件变量


mutex mutexThreadAState;
// bool bIsThreadABlocking;
bool bIsABOK;                       // 用于这两个交换数据的条件变量


// 下面是线程
void threadARun(void);
void threadBRun(void);

bool isKeepRun(void);
bool isBAOK(void);                  // 要求读取之后自动复位
bool isABOK(void);                  // 同上





// 主函数
int main(int argc, char* argv[])
{
    cout<<"Producer and consumer test."<<endl;
    cout<<"Conplied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    // 先设置都允许所有线程运行
    {
        std::unique_lock <std::mutex> lock(mutexKeepRun);
        keepRun=true;
    }

    // 开启B线程
    cout<<endl;
    cout<<"[Main Thread] Thread B stating ..."<<endl;
    thread threadB(threadBRun);

    // 开启A线程
    cout<<endl;
    cout<<"[Main Thread] Thread A stating ..."<<endl;
    thread threadA(threadARun);

    cout<<endl;
    int duration=100;
    cout<<"[Main Thread] Wait for "<<duration<<" seconds ..."<<endl;
    std::this_thread::sleep_for(std::chrono::seconds(duration));
    cout<<endl;
    cout<<"[Main Thread] Stopping thread A and B ..."<<endl;

    {
        std::unique_lock <std::mutex> lock(mutexKeepRun);
        keepRun=false;
    }
    
    threadA.join();
    threadB.join();

    cout<<endl;
    cout<<"[Main Thread] All done, programe terminated."<<endl;

    
    return 0;
}

void threadBRun(void)
{
    cout<<"[Thread B] Running."<<endl;
    ProductObject product;
    double time_s;
    bool loopFlag=true;

    {
        std::unique_lock <std::mutex> lock(mutexThreadBState);
        bIsBAOK=false;
    }

    while(loopFlag)
    // 线程B只做一件事情,接受A的半成品
    {
        // 等待线程A的通知
        while(!isABOK())
        {
            if(!isKeepRun())
            {
                loopFlag=false;
                cout<<"[Thread B] Tying to exit."<<endl;
                break;
            }
        }
        {
            std::unique_lock <std::mutex> lock(mutexAB);
            cout<<"[Thread B] Get Product #"<<firstProduct.id<<endl;
            // 模拟对数据的转存,在此期间需要保持锁住的状态
            product=firstProduct;
        }

        // 接下来的处理过程就可以暂时不加锁        
        product.content = product.content + string(" Thread B;");
        // 模拟处理过程所需要耗费的时间
        time_s=1+rand()*1.0f/RAND_MAX;;
        this_thread::sleep_for(std::chrono::milliseconds((int)(time_s*1000)));

        // 准备设置信号,上锁
        {
            std::unique_lock <std::mutex> lock(mutexBA);
            secondProduct=product;
        }

        cout<<"[Thread B] Product processd complete, time cost "<<time_s<<" s."<<endl;
        
        {
            std::unique_lock <std::mutex> lock(mutexThreadBState);
            bIsBAOK=true;
        }
    }

    cout<<"[Thread B] Stopped."<<endl;
}

void threadARun(void)
{
    cout<<"[Thread A] Running."<<endl;
    double time_s;
    size_t id_cnt=0;

    bool loopFlag=true; // 通过改动这个变量控制外层循环的退出

    {
        std::unique_lock <std::mutex> lock(mutexThreadAState);
        bIsABOK=false;
    }

    // 线程A则需要进行三件事情
    while(loopFlag)
    {
        // 第一件事情: 产生最初的产品
        if(!isKeepRun())  break;
        else
        {
            std::unique_lock <std::mutex> lock(mutexAB);
            firstProduct.content=string("List: Thread A;");
            firstProduct.id=id_cnt++;
            // 一个随机的延时来模拟生成过程中所耗费的时间
            time_s=1+rand()*1.0f/RAND_MAX;
            this_thread::sleep_for(std::chrono::milliseconds((int)(time_s*1000)));
        }
        cout<<"[Thread A] Generate product #"<<firstProduct.id<<" ok, time cost "<<time_s<<"s."<<endl;
        // ok,生成完成,线程B可以拿去搞事情了
        {
            std::unique_lock <std::mutex> lock(mutexThreadAState);
            bIsABOK=true;
        }

        // 第二件事情: 进行中间操作,这里也是使用线程的延时来进行模拟
        // 这里不用检查是否停止当前线程,否则可能会造成线程B无法结束
        time_s=1+rand()*1.0f/RAND_MAX;
        this_thread::sleep_for(std::chrono::milliseconds((int)(time_s*1000)));
        cout<<"[Thread A] Some work cost "<<time_s<<" s."<<endl;

        // 第三件事情: 拿到B已经处理完的东西,继续搞事情
        // 首先需要等待B完成
        while(!isBAOK())
        {
            // 循环过程中检查是否该关掉当前线程了
            if(!isKeepRun())
            {
                loopFlag=false;
                cout<<"[Thread A] Tring to exit"<<endl;
                break;
                // 其实也没有太好的办法退出
            }
        }
        // 确定B也已经完事了
        {
            // B已经完事了,那么我们直接拿就行了
            std::unique_lock <std::mutex> lock(mutexBA);
            cout<<"[Thread A] Get Product #"<<secondProduct.id<<"."<<endl;
            // 模拟对数据的转存,在此期间需要保持锁住的状态
            finalProduct.content=secondProduct.content+" Thread A final.";
            finalProduct.id=secondProduct.id;
        }
        // 随机延时
        time_s=1+rand()*1.0f/RAND_MAX;
        this_thread::sleep_for(std::chrono::milliseconds((int)(time_s*1000)));
        cout<<"[Thread A] Product #"<<finalProduct.id<<" processed complete, cost "<<time_s<<" s, content: "<<finalProduct<<endl;
    }

    cout<<"[Thread A] Stop."<<endl;
}


bool isKeepRun(void)
{
    std::unique_lock <std::mutex> lock(mutexKeepRun);
    return keepRun;
}

bool isBAOK(void)
{
    std::unique_lock <std::mutex> lock(mutexThreadBState);
    bool res=bIsBAOK;
    if(bIsBAOK) bIsBAOK=false;
    return res;
}

bool isABOK(void)
{
    std::unique_lock <std::mutex> lock(mutexThreadAState);
    bool res=bIsABOK;
    if(bIsABOK) bIsABOK=false;
    return res;

}