#include<condition_variable>
#include<mutex>
#include<thread>
#include<iostream>
#include<queue>
#include<chrono>
 
/*
    关键点是：
    1. wait()函数的内部实现是：先释放了互斥量的锁，然后阻塞以等待条件为真；
    2. notify系列函数需在unlock之后再被调用。
    3. 套路是：
        a. A线程拿住锁，然后wait,此时已经释放锁，只是阻塞了在等待条件为真；
        b. B线程拿住锁，做一些业务处理，然后令条件为真，释放锁，再调用notify函数；
        c. A线程被唤醒，接着运行。
*/
 
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>




#include <iostream>                // std::cout
#include <thread>                // std::thread
#include <mutex>                // std::mutex, std::unique_lock
#include <condition_variable>    // std::condition_variable
 
std::mutex mtx; // 全局互斥锁.
std::condition_variable cv; // 全局条件变量.
bool ready = false; // 全局标志位.
 
void do_print_id(int id)
{
    std::unique_lock <std::mutex> lck(mtx);
    while (!ready) // 如果标志位不为 true, 则等待...
        cv.wait(lck); // 当前线程被阻塞, 当全局标志位变为 true 之后,
    // 线程被唤醒, 继续往下执行打印线程编号id.
    std::cout << "thread " << id << '\n';
}
 
void go()
{
    std::unique_lock <std::mutex> lck(mtx);
    ready = true; // 设置全局标志位为 true.
    cv.notify_all(); // 唤醒所有线程.
}
 
int main()
{
    std::thread threads[10];
    // spawn 10 threads:
    for (int i = 0; i < 10; ++i)
        threads[i] = std::thread(do_print_id, i);
 
    std::cout << "10 threads ready to race...\n";
    go(); // go!
 
  for (auto & th:threads)
        th.join();
 
    return 0;
}

/*
std::mutex m;
std::condition_variable cv;
std::string data;
bool ready = false;
bool processed = false;
 
void worker_thread()
{
    {
        // Wait until main() sends data
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, []{return ready;});
    }
 
    {
        std::unique_lock<std::mutex> lk(m);
        // after the wait, we own the lock.
        std::cout << "Worker thread is processing data\n";
        data += " after processing";
     
        // Send data back to main()
        processed = true;
        std::cout << "Worker thread signals data processing completed\n";
    }
    
    cv.notify_one();
}
 
int main()
{
    std::thread worker(worker_thread);
 
    data = "Example data";
    // send data to the worker thread
    {
        std::unique_lock<std::mutex> lk(m);
        ready = true;
        std::cout << "main() signals data ready for processing\n";
    }
    cv.notify_one();
 
    // wait for the worker
    {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, []{return processed;});
    }
    
    std::cout << "Back in main(), data = " << data << '\n'; 
    worker.join();
}
*/

/*
int main(){
	std::queue<int>products;          //产品队列
 
	std::mutex m;
	std::condition_variable cond_var; //条件变量

	bool notifid = false;             //通知标志
	bool done = false;                //消费者进程结束标志
 
	std::thread producter( [&](){     //捕获互斥锁
 
		for(int i=1; i<10; ++i){
			std::this_thread::sleep_for(std::chrono::seconds(1)); //等待1S
 
			std::unique_lock<std::mutex>lock(m);        //创建互斥量 保证下列操作不被打断
			std::cout<<"producting "<<i<<std::endl;
			products.push(i);
			notifid = true;                             
			cond_var.notify_one();						//通知另一线程
		}
		done = true;                      //生产结束
		cond_var.notify_one();
	});
 
 
	std::thread consumer( [&](){
		while(!done){
			std::unique_lock<std::mutex>lock(m);
			while(!notifid){
				cond_var.wait(lock);             //通过调用 互斥量 来等待 条件变量
			}
			while(!products.empty()){
				std::cout<<"consumer "<<products.front()<<std::endl;
				products.pop();
			}
			notifid = false;
		}
	});
	producter.join();
	consumer.join();
	return 0;
}

*/
