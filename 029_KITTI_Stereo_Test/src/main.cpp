#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void prefilterXSobel(const cv::Mat& src, cv::Mat& dst, int ftzero);

template <typename T> void filterSpecklesImpl(cv::Mat& img, int newVal, int maxSpeckleSize, int maxDiff, cv::Mat& _buf);


int main(int argc, char* argv[])
{
    cout<<"Kitti Streo Test."<<endl;
    cout<<"Complied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    if(argc!=4)
    {
        cout<<"Usage: "<<argv[0]<<" img_left img_right img_dis"<<endl;
        return 1;
    }

    // 读入左右双目图像
    Mat imgLeft =imread(argv[1],IMREAD_GRAYSCALE);
    Mat imgRight=imread(argv[2],IMREAD_GRAYSCALE);

    if(imgLeft.empty())
    {
        cout<<"Error: img_left "<<argv[1]<<" is empty!"<<endl;
        return 2;
    }

    if(imgRight.empty())
    {
        cout<<"Error: img_left "<<argv[1]<<" is empty!"<<endl;
        return 2;
    }

    imshow("img_left",imgLeft);
    imshow("img_right",imgRight);

    waitKey(1);

    Mat imgLefted,imgRighted;

    

    // 最小视差
    int mindisparity = 0;
    // 视差搜索范围长度
	int ndisparities = 64;  
    // SAD代价计算窗口大小
	int SADWindowSize = 11; 
	//SGBM
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);

    // 能量函数参数
	int P1 = 8 * imgLeft.channels() * SADWindowSize* SADWindowSize;
    // 能量函数参数
	int P2 = 32 * imgRight.channels() * SADWindowSize* SADWindowSize;

    // 下面就是各种配置了
	sgbm->setP1(P1);
	sgbm->setP2(P2);
	sgbm->setPreFilterCap(15);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleRange(2);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setDisp12MaxDiff(1);
	//sgbm->setMode(cv::StereoSGBM::MODE_HH);

    // 对原始图像进行预处理
    // imgLeft.copyTo(imgLefted);
    // imgRight.copyTo(imgRighted);
    // prefilterXSobel(imgLeft, imgLefted, sgbm->getPreFilterCap());
    // prefilterXSobel(imgRight, imgRighted, sgbm->getPreFilterCap());

    // imshow("img_left",imgLefted);
    // imshow("img_right",imgRighted);

    // waitKey(0);

    // 
    Mat disp;
	sgbm->compute(imgLeft, imgRight, disp);
	disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值
	Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示
	normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);

    // display
    imshow("dis",disp8U);

    waitKey(0);

	imwrite(argv[3], disp8U);

    return 0;
}

void prefilterXSobel(const cv::Mat& src, cv::Mat& dst, int ftzero)
{
    int x, y;
    const int OFS = 256 * 4, TABSZ = OFS * 2 + 256;
    uchar tab[TABSZ];
    cv::Size size = src.size();

    for (x = 0; x < TABSZ; x++)
        tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero * 2 : x - OFS + ftzero);
    uchar val0 = tab[0 + OFS];

    for (y = 0; y < size.height - 1; y += 2)
    {
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow0 = y > 0 ? srow1 - src.step : size.height > 1 ? srow1 + src.step : srow1;
        const uchar* srow2 = y < size.height - 1 ? srow1 + src.step : size.height > 1 ? srow1 - src.step : srow1;
        const uchar* srow3 = y < size.height - 2 ? srow1 + src.step * 2 : srow1;
        uchar* dptr0 = dst.ptr<uchar>(y);
        uchar* dptr1 = dptr0 + dst.step;

        dptr0[0] = dptr0[size.width - 1] = dptr1[0] = dptr1[size.width - 1] = val0;
        x = 1;
        for (; x < size.width - 1; x++)
        {
            int d0 = srow0[x + 1] - srow0[x - 1], d1 = srow1[x + 1] - srow1[x - 1],
                d2 = srow2[x + 1] - srow2[x - 1], d3 = srow3[x + 1] - srow3[x - 1];
            int v0 = tab[d0 + d1 * 2 + d2 + OFS];
            int v1 = tab[d1 + d2 * 2 + d3 + OFS];
            dptr0[x] = (uchar)v0;
            dptr1[x] = (uchar)v1;
        }
    }

    for (; y < size.height; y++)
    {
        uchar* dptr = dst.ptr<uchar>(y);
        x = 0;
        for (; x < size.width; x++)
            dptr[x] = val0;
    }
}


typedef cv::Point_<short> Point2s;
template <typename T> void filterSpecklesImpl(cv::Mat& img, int newVal, int maxSpeckleSize, int maxDiff, cv::Mat& _buf)
{
    using namespace cv;

    int width = img.cols, height = img.rows, npixels = width*height;
    size_t bufSize = npixels*(int)(sizeof(Point2s) + sizeof(int) + sizeof(uchar));
    if (!_buf.isContinuous() || _buf.empty() || _buf.cols*_buf.rows*_buf.elemSize() < bufSize)
        _buf.create(1, (int)bufSize, CV_8U);

    uchar* buf = _buf.ptr();
    int i, j, dstep = (int)(img.step / sizeof(T));
    int* labels = (int*)buf;
    buf += npixels * sizeof(labels[0]);
    Point2s* wbuf = (Point2s*)buf;
    buf += npixels * sizeof(wbuf[0]);
    uchar* rtype = (uchar*)buf;
    int curlabel = 0;

    // clear out label assignments
    memset(labels, 0, npixels * sizeof(labels[0]));

    for (i = 0; i < height; i++)
    {
        T* ds = img.ptr<T>(i);
        int* ls = labels + width*i;

        for (j = 0; j < width; j++)
        {
            if (ds[j] != newVal)   // not a bad disparity
            {
                if (ls[j])     // has a label, check for bad label
                {
                    if (rtype[ls[j]]) // small region, zero out disparity
                        ds[j] = (T)newVal;
                }
                // no label, assign and propagate
                else
                {
                    Point2s* ws = wbuf; // initialize wavefront
                    Point2s p((short)j, (short)i);  // current pixel
                    curlabel++; // next label
                    int count = 0;  // current region size
                    ls[j] = curlabel;

                    // wavefront propagation
                    while (ws >= wbuf) // wavefront not empty
                    {
                        count++;
                        // put neighbors onto wavefront
                        T* dpp = &img.at<T>(p.y, p.x); //current pixel value
                        T dp = *dpp;
                        int* lpp = labels + width*p.y + p.x; //current label value

                        //bot
                        if (p.y < height - 1 && !lpp[+width] && dpp[+dstep] != newVal && std::abs(dp - dpp[+dstep]) <= maxDiff)
                        {
                            lpp[+width] = curlabel;
                            *ws++ = Point2s(p.x, p.y + 1);
                        }
                        //top
                        if (p.y > 0 && !lpp[-width] && dpp[-dstep] != newVal && std::abs(dp - dpp[-dstep]) <= maxDiff)
                        {
                            lpp[-width] = curlabel;
                            *ws++ = Point2s(p.x, p.y - 1);
                        }
                        //right
                        if (p.x < width - 1 && !lpp[+1] && dpp[+1] != newVal && std::abs(dp - dpp[+1]) <= maxDiff)
                        {
                            lpp[+1] = curlabel;
                            *ws++ = Point2s(p.x + 1, p.y);
                        }
                        //left
                        if (p.x > 0 && !lpp[-1] && dpp[-1] != newVal && std::abs(dp - dpp[-1]) <= maxDiff)
                        {
                            lpp[-1] = curlabel;
                            *ws++ = Point2s(p.x - 1, p.y);
                        }
                        

                        // pop most recent and propagate
                        // NB: could try least recent, maybe better convergence
                        p = *--ws;
                    }

                    // assign label type
                    if (count <= maxSpeckleSize)   // speckle region
                    {
                        rtype[ls[j]] = 1;   // small region label
                        ds[j] = (T)newVal;
                    }
                    else
                        rtype[ls[j]] = 0;   // large region label
                }
            }
        }
    }
}