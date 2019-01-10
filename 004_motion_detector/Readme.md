# 使用几种经典算法进行运动目标检测实验



# 一、功能简述

使用简单背景减除法、帧差法（以及三帧差法）和混合高斯模型法进行运动物体检测的尝试。

使用CDW-2014数据集中的baseline。

调用：

```
MotionDetector data_path mode
```

其中：
- data_path 数据集目录
- mode 选择的算法，如下：
  - 0 简单背景减除法
  - 1 帧差法
  - 2 三帧差法
  - 3 自己写的GMM算法
  - 4 opencv提供的MOG算法（效果非常不好）


# 二、编译

## 2.1 依赖

### 必要依赖

- OpenCV (实验中使用的是3.3.1版本)

### 可选依赖

- doxygen (实验中使用的是 1.8.11 版本)

## 2.2 开发平台和工具

- Ubuntu 16.04
- VScode

## 2.3 编译工具链

- cmake 3.5.1
- g++ 5.4.0

## 2.4 其他

无

# 五、程序结构

主函数负责总体功能模块调度。

- DataReader类
  负责读取CDW-2014数据集中的数据，并且获取帧数、帧画面的长度、宽度等信息。
- MotionDetector_base
  是运动检测器的基类.
- MotionDetector_DiffBase :MotionDetector_base
  是基于差分图像的运动检测器的基类。封装了后端的形态学操作，提供了中间图像的数据接口；计算差分图像的任务则交给子类来实现。
- MotionDetector_backsub:MotionDetector_DiffBase
  简单背景减除法
- MotionDetector_framesub:MotionDetector_DiffBase
  帧差法（两帧的那种）
- MotionDetector_3framesub:MotionDetector_framesub
  三帧差法
- Motiondetector_GMM:MotionDetector_DiffBase
  使用opencv的MOG2算法，但可能是参数设置的原因，效果不好。
- Motiondetector_GMM2:MotionDetector_DiffBase
  自己根据GMM模型编写的GMM算法。


# 四、注意事项

无

# 五、参考


- [【OpenCV学习笔记】三十九、运动物体检测(一)](https://blog.csdn.net/abc8730866/article/details/70170267)
- [前景检测算法_3(GMM)](https://www.cnblogs.com/tornadomeet/archive/2012/06/02/2531565.html)