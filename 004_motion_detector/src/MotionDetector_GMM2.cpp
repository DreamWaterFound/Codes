/**
 * @file MotionDetector_GMM2.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 自己实现的GMM前景检测
 * @version 0.1
 * @date 2019-01-08
 * 
 * @copyright Copyright (c) 2019
 * 
 */


#include "MotionDetector_GMM2.h"

using namespace std;
using namespace cv;

//构造函数
MotionDetector_GMM2::MotionDetector_GMM2()
{
    resetDetector();
}

//析构函数
MotionDetector_GMM2::~MotionDetector_GMM2()
{
    ;
}

cv::Mat MotionDetector_GMM2::calcuDiffImg(cv::Mat frame)
{
    cvtColor(frame, frame, CV_BGR2GRAY);
    
    
    //根据当前的帧计数来选择不同的操作
    mnFrameCnt++;
    
    if(mnFrameCnt==1)
    {
        //对第一帧执行特殊的初始化操作
        gmmInit(frame);
        gmmDealFirstFrame(frame);
    }
    else if(mnFrameCnt<=mnLearnFrameNumber)
    {
        //如果现在在训练的帧中
        gmmTrain(frame);

        if(mnFrameCnt==mnLearnFrameNumber)
        {
            //如果当前是训练的最后一帧,那么还应该需要计算适应数值
            gmmCalFitNum(frame);
        }
    }
    else
    {
        //说明现在是需要进行识别操作了
        gmmTest(frame);
    }

    return mmGMMMask;
}

//初始化所有参数
void MotionDetector_GMM2::resetDetector(void)
{
    //由于高斯模型们的参数依赖于帧的大小,所以这里并不能够直接初始化

    mfLearnRate=DEFAULT_GMM_LEARN_ALPHA;
    mfThreshod=DEFALUT_GMM_THRESHOD_SUMW;
    mnLearnFrameNumber=DEFAULT_END_FRAME;

    mnFrameCnt=0;

    resetDetector_base();
}

//初始化GMM模型的参数
void MotionDetector_GMM2::gmmInit(cv::Mat img)
{
    for(int i=0;i<GMM_MAX_COMPONT;i++)
    {
        mmWeight[i]=cv::Mat(img.rows,img.cols,CV_32FC1,0.0);
        mmU[i]=cv::Mat(img.rows,img.cols,CV_8UC1,-1);
        mmSigma[i]=cv::Mat(img.rows,img.cols,CV_32FC1,0.0);
    }

    mmFitNum=cv::Mat::zeros(img.rows,img.cols,CV_8UC1);
    mmGMMMask=cv::Mat::zeros(img.rows,img.cols,CV_8UC1);
}

//GMM处理第一帧
void MotionDetector_GMM2::gmmDealFirstFrame(cv::Mat img)
{
    //遍历图像中的每一个像素
    for(int x=0;x<img.rows;x++)
    {
        for(int y=0;y<img.cols;y++)
        {
            /** 对于图像中的每一个元素,进行下面的操作: */
            /** 首先无论我们确定的高斯模型的个数是多少,但是我们总可以确定的是,至少有一个高斯模型.\n
             * 所以,赋值为初始值 1.0 
             */ 
            //mmWeight[0].at<float>(x,y)=1.0;
            W(0,x,y)=1.0;
            /** 均值就是当前像素的灰度值 */
            //mmU[0].at<unsigned char>(x,y)=img.at<unsigned char>(x,y);
            U(0,x,y)=img.at<unsigned char>(x,y);
            /** 方差则根据OpenCV库中的源码,选择为15. */
            //mmSigma[0].at<float>(x,y)=15.0;
            Sigma(0,x,y)=15.0;

            /** 然后对于其他的高斯分布模型,也要进行相同的操作. */
            for(int i=1;i<GMM_MAX_COMPONT;i++)
            {
                W(i,x,y)=0.0;
                //TODO  对于这个我现在也还是不是非常明白
                U(i,x,y)=-1;
                Sigma(i,x,y)=15.0;
            }//对于其他的高斯分布模型,也要进行相同的操作
        }//见下
    }//遍历图像中的每一个像素
}

//GMM模型的训练函数
void MotionDetector_GMM2::gmmTrain(cv::Mat img)
{
    //遍历图像中的每一个像素
    for(int x=0;x<img.rows;x++)
    {
        for(int y=0;y<img.cols;y++)
        {
            //高斯模型匹配失败计数
            int fallFitCnt=0;

            //开始遍历这个像素下的每个高斯模型
            for(int k=0;k<GMM_MAX_COMPONT;k++)
            {
                /** 对于图像中的每一个像素的每一个高斯模型,执行下面的操作: <ul>*/
                //CHECKPOINT
                /** <li> 1.计算当前帧图像这个像素的灰度和当前高斯模型均值的差,及其平方: </li> 
                 * \f$ \Delta=\mathbf{I}(x,y)-u_{k}(x,y) \f$ \n
                 * \f$ \Delta^2 \f$
                */
                int err=(int)IMG(x,y)-(int)U(k,x,y);
                long err2=err*err;

                /** <li> 2.检查是否匹配当前的高斯模型, 如果下面的不等式成立,说明满足: </li> \n
                 * \f$ |\mathbf{I}(x,y)-U_{k}(x,y)| < 3\Sigma_{k}(x,y) \f$ <ul>                * 
                */
                if(err<3.0*Sigma(k,x,y))
                {
                    /** <li> 如果能够匹配,那么就要安装下面的公式更新权值: </li> */

                    /** \f$ w_{k}(x,y)=w_{k}(x,y)+\alpha(1-w_{k}(x,y)) \f$ \n
                     * 其中 \f$ \alpha \f$ 是学习速率.
                    */
                    W(k,x,y)=W(k,x,y)+mfLearnRate*(1-W(k,x,y));
                    
                    /** \f$  u_{k}(x,y)=u_{k}(x,y)+\alpha/{w_{k}(x,y)\Delta}  \f$ */
                    U(k,x,y)=U(k,x,y)+(mfLearnRate/W(k,x,y))*err;

                    /** \f$ \Sigma_{k}(x,y)=\Sigma_{k}(x,y)+\alpha/{w_{k}(x,y)\cdot\Delta^2-\Sigma_{k}(x,y)} \f$ */
                    Sigma(k,x,y)=Sigma(k,x,y)+(mfLearnRate/W(k,x,y))*(err2-Sigma(k,x,y));                   
                }
                else
                {
                    /** <li> 如果不能够匹配,按照下面的公式来更新权值, 同时失败计数累加</li> */

                    /** /f$ w_{k}(x,y)=w_{k}(x,y)-\alpha w_{k}(x,y) /f$ */
                    W(k,x,y)=W(k,x,y)+mfLearnRate*(0-W(k,x,y));
                    fallFitCnt++;
                }//更新所有的高斯模型
                /** </ul> */

                /** <li> 3.对所有高斯模型按照指标 w/sigma 进行排序.</li> */
                //这里使用的是冒泡排序法

                float tempW,tempSigma;
                unsigned char tempU;

                for(int kk=0;kk<GMM_MAX_COMPONT;kk++)
                {
                    for(int rr=kk;rr<GMM_MAX_COMPONT;rr++)
                    {
                        //判断是否符合冒泡条件
                        if(W(rr,x,y)/Sigma(rr,x,y) < W(kk,x,y)/Sigma(kk,x,y))
                        {
                            //交换权值
                            tempW=W(rr,x,y);
                            W(rr,x,y)=W(kk,x,y);
                            W(kk,x,y)=tempW;

                            //交换均值
                            tempU=U(rr,x,y);
                            U(rr,x,y)=U(kk,x,y);
                            U(kk,x,y)=tempU;

                            //交换方差
                            tempSigma=Sigma(rr,x,y);
                            Sigma(rr,x,y)=Sigma(kk,x,y);
                            Sigma(kk,x,y)=tempSigma;
                         }//判断是否符合冒泡交换条件
                    }//见下
                }//对所有高斯模型进行冒泡排序

                /** <li> 4.现在要进行检查.如果没有满足条件的高斯,那么就要重开开始一个高斯分布.条件有下面两个:</li> <ul>*/
                    /** <li> 1.刚才的匹配过程中,当前的像素点没有任何一个能够匹配上的高斯模型 </li>
                     *  <li> 2.经过刚刚的排序之后所得到的,比值最小的高斯分布的对应的权值已经为0了,说明这个分布已经是被彻底遗弃了 <li> </ul> \n
                     * 只有上面的两个条件同时满足的时候,才能够对这个像素的每个高斯分布进行下面的操作. <ul> */
                if(fallFitCnt==GMM_MAX_COMPONT &&
                    0 == W(GMM_MAX_COMPONT-1,x,y)
                  )
                {
                    //遍历这个像素的每个高斯分布
                    for(int h=0;h<GMM_MAX_COMPONT;h++)
                    {
                        /** <li> 1. 在这个像素的每个高斯分布中找到第一个权值为0的分布 </li>  */
                        if(0 == W(h,x,y))
                        {
                            /** <li> 2. 根据论文中的参数设计,这里直接将它的权值设置成为学习率 </li>  */
                            W(h,x,y)=mnLearnFrameNumber;
                            /** <li> 3. 均值设置成为当前像素的灰度 </li> */
                            U(h,x,y)=(unsigned char)IMG(x,y);
                            /** <li> 4. 方差则设置成为默认的15 </li> */
                            Sigma(h,x,y)=15.0;

                            /** <li> 5. 然后还要从最开始的分布,到这个第一次出现0权值的分布之间的所有分布,都要按照上面的规则重新设置他们的权值 </li>
                             * \n 权值更新为:
                             * \n \f$ w_{k}(x,y)=w_{k}(x,y)*(1-\alpha) \f$
                             * \n 但是均值和方差是不变的
                            */
                            for(int q=0;q<GMM_MAX_COMPONT;q++)
                            {
                                W(q,x,y)*=1-mnLearnFrameNumber;
                            }

                            //NOTICE 找到第一个权值不为0的即可
                            /** </ul> */
                        }//找到了一个权值为0的高斯分布
                    }//遍历这个像素的每个高斯分布
                }//判断是否需要重开一个高斯分布
                
                /** <li> 5. 如果满足条件一不满足条件二,那么就要重设这个比值最弱的高斯分布的均值和方差.</li> */ 
                else if(fallFitCnt==GMM_MAX_COMPONT &&
                        W(GMM_MAX_COMPONT-1,x,y !=0) )
                {
                    //那么就要将最后的那个比值最弱的高斯分布重新设置为默认的参数    
                    U(GMM_MAX_COMPONT-1,x,y)=IMG(x,y);
                    Sigma(GMM_MAX_COMPONT-1,x,y)=15.0;
                }//判断是否需要重设一个高斯分布
                /** </ul> */
            }//遍历这个像素下的每个高斯模型
        }//见下
    }//遍历图像中的每一个像素
}

//根据当前帧确定每个像素最适合使用的高斯模型是多少个
void MotionDetector_GMM2::gmmCalFitNum(cv::Mat img)
{
    //开始遍历图像中的每一个像素
    for(int x=0;x<img.rows;x++)
    {
        for(int y=0;y<img.cols;y++)
        {
            /** 对于图像中的每一个像素的每一个高斯模型: */

            float sum=0.0f;
            for(unsigned char a=0;a<GMM_MAX_COMPONT;a++)
            {
                /** 累加这个高斯模型的权值 */
                sum+=W(a,x,y);
                /** 如果发现累加到当前的高斯分布的时候,权值的累加和已经达到或者超过预定的阈值了 */
                if(sum>=DEFALUT_GMM_THRESHOD_SUMW)
                {
                    /** 那么就说说明使用这些个高斯分布就已经足够了 */
                    //下面的a+1是因为在循环中的a是从0开始的
                    mmFitNum.at<unsigned char>(x,y)=a+1;
                    break;
                }//查看是否已经超过预定的阈值
            }//遍历这个像素的每个高斯分布
        }//见下
    }//遍历图像中的每一个像素
}

//根据输入的图像进行测试
void MotionDetector_GMM2::gmmTest(cv::Mat img)
{
    //还是遍历所有的图像像素
    for(int x=0;x<img.rows;x++)
    {
        for(int y=0;y<img.cols;y++)
        {
            /** 对于所有的图像像素,都只选择那几个最有效的模型进行计算 */
            //之所以这个初始化写在这里是因为在循环完成之后还是要用到
            unsigned char a=0;
            for(;a<mmFitNum.at<unsigned char>(x,y);a++)
            {
                if(abs(IMG(x,y)-U(a,x,y)) < 
                    (unsigned char)(2.5*(Sigma(a,x,y))))
                {
                    //到这里就可以得到判断,只要和其中的一个高斯分布的方差比较接近,那么就可以认为是背景
                    mmGMMMask.at<unsigned char>(x,y)=1;
                    break;
                }//判断是否是背景
            }//遍历所有的有效模型

            //如果上述都不是,那么就可以认为当前的这个像素是前景
            if(a==mmFitNum.at<unsigned char>(x,y))
            {
                mmGMMMask.at<unsigned char>(x,y)=255;
            }

        }//见下
    }//遍历所有的图像像素
}


