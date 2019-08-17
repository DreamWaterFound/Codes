# 044 Producer-consumer_model

## 目的和想法

对多线程过程中"生产者-消费者"模型的实操.准备使用C++11中的条件变量来实现.

其实原本的互斥锁也可以,但是效率不高;目前的多线程之间的模型大致如下:

A --> B --> A

我需要其中数据A的到来是我无法控制的;B只用来处理A的结果,也许大部分时间B是空闲的,但是我要求B能够在线程A有数据产生的时候**及时**地去处理它;并且我不希望在B等待A数据的时候通过轮询的方式,因为这样会在等待过程中就消耗大量的计算资源.当A把B需要的数据生产完成之后,它还需要去进行别的工作,然后在某个步骤的时候又会开始需要使用B处理之后的数据.类似地,当B生产完成之后,A线程不一定有空能够接受这个数据,但是如果A有空了,就要**及时**地处理B的数据;如果发生等待,我也不希望A使用轮询的方式,原因也是因为这样操作会消耗大量的计算资源.所以互斥量的方式在这里不太适合.

目前找到的可能是比较好的解决方式是这种条件变量.试试吧,试试又没有啥大不了,对不对.

## 结果

嗯，很鲜明，对于轮询的方式还是很吃CPU的（尽管目前的线程中有相当一部分时间是在睡眠状态）：

![轮询方式](https://github.com/DreamWaterFound/Codes/blob/master/044_Producer-consumer_model/doc/contrast.gif)

而使用条件变量的方案，线程大部分时间都是在睡眠中，因此几乎不怎么占用CPU的计算资源：

![条件变量方式](https://github.com/DreamWaterFound/Codes/blob/master/044_Producer-consumer_model/doc/condition.gif)


## Ref

https://blog.csdn.net/AXuan_K/article/details/51972012

这篇不错

https://www.2cto.com/kf/201506/411327.html
