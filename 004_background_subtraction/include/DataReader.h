/**
 * @file DataReader.h
 * @author guoqing (1337841346@qq.com)
 * @brief 数据库阅读器类的声明
 * @version 0.1
 * @date 2019-01-05
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "common.h"


/**
 * @brief 读取数据集CDW-2014的类
 * 
 */
class DataReader
{
public:
    /**
     * @brief Construct a new Data Reader object
     * 
     */
    DataReader();
    
    /**
     * @brief Destroy the Data Reader object
     * 
     */
    ~DataReader();

    /**
     * @brief 打开图像序列的
     * 
     * @param[in] path  图像序列的位置
     * @return true     打开成功
     * @return false    打开失败
     */
    bool openSeq(const char* path);

    /**
     * @brief 关闭序列
     * @detials 其实主要是关闭读取序列的指针
     */
    void closeSeq(void);

    /**
     * @brief 获取视频序列中的总帧数
     * @return int 总帧数。如果为0说明要么没有，要么数据集打开失败。
     */
    inline int getTotalFrames(void)
    {
        return mnFrameLength;
    }

    /**
     * @brief Get the New Frame 
     * 
     * @param[out] img  新的一帧的图像
     * @return true     获取成功
     * @return false    获取失败（一般是因为数据集没有正确打开或者是已经是最后一帧了）
     */
    bool getNewFrame(cv::Mat &img);

    /**
     * @brief 重设读取帧的指针
     * 
     */
    void ResetFramePos(void);

    /**
     * @brief 获取当前帧的位置
     * 
     * @return int 当前帧的位置
     */
    int getCurrFramePos(void);

    /**
     * @brief 获取视频的播放速率
     * @notes 但是目前还没有比较好的确定图像序列的帧率的办法，所以目前
     * 采取的措施还是要靠人工给定
     * 
     * @return int 帧率
     */
    inline int getFPS(void)
    {
        return mnFPS;
    }

private:

    ///图像帧率
    int mnFPS;      ///<图像帧率
    ///帧的大小
    cv::Size mFrameSize;    
    ///帧的总长度
    int mnFrameLength;
    ///当前帧的位置
    int mnCurFramePos;
    ///数据集的路径
    string data_path;
};