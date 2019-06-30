#include <iostream>
#include <string>
#include <vector>

#include "yolact.hpp"
#include "tools/TUMRGBD_DataReader.hpp"

using namespace std;

int main(int argc, char* argv[])
{
    cout<<"Test optimized YOLACT c++ interface for TUM RGBD test."<<endl;
    cout<<"Complied at "<<__TIME__<<" "<<__DATE__<<"."<<endl;
    
    if(argc!=8)
    {
        cout<<"Usage: "<<argv[0]<<" python_env_pkgs_path python_moudle_path init_python_function_name eval_python_function_name trained_model_path TUM_RGBD_PATH ASSOCIATE_PATH"<<endl;
        return 0;
    }

    string strPyEnvPkgsPath(argv[1]);
    string strPyMoudlePathAndName(argv[2]);
    string strInitPyFunctionName(argv[3]);
    string strEvalPyfunctionName(argv[4]);
    string strTrainedModelPath(argv[5]);
    string strEvalImagePath(argv[6]);

    DataReader::TUM_DataReader reader(argv[6],argv[7]);

    YOLACT::YOLACT yolact_net(
        strPyEnvPkgsPath,
        strPyMoudlePathAndName,
        strInitPyFunctionName,
        strEvalPyfunctionName,
        strTrainedModelPath,
        0.3,10);

    if(yolact_net.isInitializedResult())
    {
        cout<<"Main: yolact ok."<<endl;
        cout<<"We have "<<yolact_net.getCLassNum()<<" classes."<<endl;
        // cout<<"They are:"<<endl;
        // // 注意这里获得是常值引用
        // const vector<string> classes=yolact_net.getClassNames();
        // for(size_t i=0;i<classes.size();++i)
        // {
        //     cout<<classes[i]<<endl;
        // }
    }
    else
    {
        cout<<"Main: yolact Failed. Error message:"<<endl;
        cout<<yolact_net.getErrorDescriptionString()<<endl;
        return 0;
    }

    vector<size_t> vstrClassName;
    vector<float> vdScores;
    vector<pair<cv::Point2i,cv::Point2i> > vpairBBoxes;
    vector<cv::Mat> vimgMasks;
    cv::Mat res,src;

    cv::Mat depth;
    double timeStamp;
    std::vector<double> groundTruth;

    while(reader.getNextItems(src,depth,timeStamp,groundTruth))
    {

        if(src.empty())
        {
            cout<<"Read image "<<strEvalImagePath<<" Error!"<<endl;
            return 0;
        }

        // cout<<"Eval ing..."<<endl;

        // 说明读入的图像是没有问题的，现在准备进行评估
        bool isOk=yolact_net.EvalImage(src,res,
            vstrClassName,vdScores,vpairBBoxes,vimgMasks);

        if(!isOk)
        {
            cout<<"Error occured when eval image. tips:"<<endl;
            cout<<yolact_net.getErrorDescriptionString()<<endl;
            return 0;
        }

        // cout<<"Eval ok."<<endl;

        if(res.empty())
        {
            cout<<"Error: result iamge is empty!"<<endl;
            return 0;
        }

        // 既然运行ok，那么我们就要显示结果图像了～
        cv::imshow("Result",res);
        cv::waitKey(1);
    }

    cout<<"OK."<<endl;

    return 0;

}



