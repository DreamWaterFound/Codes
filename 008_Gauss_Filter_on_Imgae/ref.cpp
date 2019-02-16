//====================================  REF  ==========================================
/**
 * @brief 产生一个高斯模板
 * 
 * @param[out] window    窗口
 * @param[in]  ksize     高斯模板的大小
 * @param[in]  sigma     高斯函数的方差
 */
void generateGaussianTemplate(double window[][11], int ksize, double sigma)
{
    static const double pi = 3.1415926;

    int center = ksize / 2; // 模板的中心位置，也就是坐标的原点

    //平方
    double x2, y2;
    //沿着x轴开始遍历计算
    for (int i = 0; i < ksize; i++)
    {   
        //在模板中,当前遍历到的位置的x坐标到"中心"距离的平方
        x2 = pow(i - center, 2);

        //遍历y轴
        for (int j = 0; j < ksize; j++)
        {
            //计算y^2
            y2 = pow(j - center, 2);
            //计算当前点出的高斯函数值
            double g = exp(-(x2 + y2) / (2 * sigma * sigma));
            g /= 2 * pi * sigma;
            //这个模板处像素的值就是这个
            window[i][j] = g;
        }
    }

    //计算缩放系数
    double k = 1 / window[0][0]; // 将左上角的系数归一化为1
    //然后对整个模板中的像素进行缩放
    for (int i = 0; i < ksize; i++)
    {
        for (int j = 0; j < ksize; j++)
        {
            window[i][j] *= k;
        }
    }
}

/**
 * @brief 实现高斯滤波器
 * 
 * @param[in] src       源图像
 * @param[in] dst       输出图像
 * @param[in] ksize     高斯核大小
 * @param[in] sigma     高斯方差
 */
void GaussianFilter(const Mat &src, Mat &dst, int ksize, double sigma)
{
    CV_Assert(src.channels() || src.channels() == 3); // 只处理单通道或者三通道图像
    const static double pi = 3.1415926;
    
    // 根据窗口大小和sigma生成高斯滤波器模板
    // 申请一个二维数组，存放生成的高斯模板矩阵
    double **templateMatrix = new double*[ksize];
    for (int i = 0; i < ksize; i++)
        templateMatrix[i] = new double[ksize];
    int origin = ksize / 2; // 以模板的中心为原点
    double x2, y2;
    double sum = 0;
    for (int i = 0; i < ksize; i++)
    {
        x2 = pow(i - origin, 2);
        for (int j = 0; j < ksize; j++)
        {
            y2 = pow(j - origin, 2);
            // 高斯函数前的常数可以不用计算，会在归一化的过程中给消去
            double g = exp(-(x2 + y2) / (2 * sigma * sigma));
            sum += g;
            templateMatrix[i][j] = g;
        }
    }
    for (int i = 0; i < ksize; i++)
    {
        for (int j = 0; j < ksize; j++)
        {
            templateMatrix[i][j] /= sum;
            cout << templateMatrix[i][j] << " ";
        }
        cout << endl;
    }
    // 将模板应用到图像中
    int border = ksize / 2;
    copyMakeBorder(src, dst, border, border, border, border, BorderTypes::BORDER_REFLECT);
    int channels = dst.channels();
    int rows = dst.rows - border;
    int cols = dst.cols - border;

    //开始遍历图像中的每个像素
    for (int i = border; i < rows; i++)
    {
        for (int j = border; j < cols; j++)
        {
            //对于图像中的每个像素
            double sum[3] = { 0 };
            for (int a = -border; a <= border; a++)
            {
                for (int b = -border; b <= border; b++)
                {
                    if (channels == 1)
                    {
                        sum[0] += templateMatrix[border + a][border + b] * dst.at<uchar>(i + a, j + b);
                    }
                    else if (channels == 3)
                    {
                        Vec3b rgb = dst.at<Vec3b>(i + a, j + b);
                        auto k = templateMatrix[border + a][border + b];
                        sum[0] += k * rgb[0];
                        sum[1] += k * rgb[1];
                        sum[2] += k * rgb[2];
                    }
                }
            }
            for (int k = 0; k < channels; k++)
            {
                if (sum[k] < 0)         //NOTCIE 一般地很少会出现这种情况吧?
                    sum[k] = 0;
                else if (sum[k] > 255)
                    sum[k] = 255;
            }
            if (channels == 1)
                dst.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
            else if (channels == 3)
            {
                Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    // 释放模板数组
    for (int i = 0; i < ksize; i++)
        delete[] templateMatrix[i];
    delete[] templateMatrix;
}


/*
REVIEW 上面的函数，感觉参考的程序中有几个问题不太对:
1. 在进行卷积的过程中,在某个像素的遍历后会覆写这个像素;这样子不会导致后面依赖于这个像素的卷积过程出现错误吗?因此感觉这个应该是有问题的;
2. dst这个图像中是保存了扩充边界的结果的;如果直接返回这个dst结果的话,滤波器输出的图像边缘也会有这个扩充的边界.这个是不是应该进行去除?
*/



//=================================================   REF   ===========================================================

// 分离的计算
void separateGaussianFilter(const Mat &src, Mat &dst, int ksize, double sigma)
{
    CV_Assert(src.channels()==1 || src.channels() == 3); // 只处理单通道或者三通道图像
    // 生成一维的高斯滤波模板
    double *matrix = new double[ksize];
    double sum = 0;
    int origin = ksize / 2;
    for (int i = 0; i < ksize; i++)
    {
        // 高斯函数前的常数可以不用计算，会在归一化的过程中给消去
        double g = exp(-(i - origin) * (i - origin) / (2 * sigma * sigma));
        sum += g;
        matrix[i] = g;
    }
    // 归一化
    for (int i = 0; i < ksize; i++)
        matrix[i] /= sum;
    // 将模板应用到图像中
    int border = ksize / 2;
    copyMakeBorder(src, dst, border, border, border, border, BorderTypes::BORDER_REFLECT);
    int channels = dst.channels();
    int rows = dst.rows - border;
    int cols = dst.cols - border;
    // 水平方向
    for (int i = border; i < rows; i++)
    {
        for (int j = border; j < cols; j++)
        {
            double sum[3] = { 0 };
            for (int k = -border; k <= border; k++)
            {
                if (channels == 1)
                {
                    sum[0] += matrix[border + k] * dst.at<uchar>(i, j + k); // 行不变，列变化；先做水平方向的卷积
                }
                else if (channels == 3)
                {
                    Vec3b rgb = dst.at<Vec3b>(i, j + k);
                    sum[0] += matrix[border + k] * rgb[0];
                    sum[1] += matrix[border + k] * rgb[1];
                    sum[2] += matrix[border + k] * rgb[2];
                }
            }
            for (int k = 0; k < channels; k++)
            {
                if (sum[k] < 0)
                    sum[k] = 0;
                else if (sum[k] > 255)
                    sum[k] = 255;
            }
            if (channels == 1)
                dst.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
            else if (channels == 3)
            {
                Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    // 竖直方向
    for (int i = border; i < rows; i++)
    {
        for (int j = border; j < cols; j++)
        {
            double sum[3] = { 0 };
            for (int k = -border; k <= border; k++)
            {
                if (channels == 1)
                {
                    sum[0] += matrix[border + k] * dst.at<uchar>(i + k, j); // 列不变，行变化；竖直方向的卷积
                }
                else if (channels == 3)
                {
                    Vec3b rgb = dst.at<Vec3b>(i + k, j);
                    sum[0] += matrix[border + k] * rgb[0];
                    sum[1] += matrix[border + k] * rgb[1];
                    sum[2] += matrix[border + k] * rgb[2];
                }
            }
            for (int k = 0; k < channels; k++)
            {
                if (sum[k] < 0)
                    sum[k] = 0;
                else if (sum[k] > 255)
                    sum[k] = 255;
            }
            if (channels == 1)
                dst.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
            else if (channels == 3)
            {
                Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    delete[] matrix;
}