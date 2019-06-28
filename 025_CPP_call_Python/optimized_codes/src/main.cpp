#include <iostream>
#include <string>
#include <vector>

#include "yolact.hpp"

using namespace std;

int main(int argc, char* argv[])
{
    cout<<"Test optimized YOLACT c++ interface."<<endl;
    cout<<"Complied at "<<__TIME__<<" "<<__DATE__<<"."<<endl;
    
    if(argc!=7)
    {
        cout<<"Usage: "<<argv[0]<<" python_env_pkgs_path python_moudle_path init_python_function_name eval_python_function_name trained_model_path image_path"<<endl;
        return 0;
    }

    string strPyEnvPkgsPath(argv[1]);
    string strPyMoudlePathAndName(argv[2]);
    string strInitPyFunctionName(argv[3]);
    string strEvalPyfunctionName(argv[4]);
    string strTrainedModelPath(argv[5]);
    string strEvalImagePath(argv[6]);

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
        cout<<"They are:"<<endl;
        // 注意这里获得是常值引用
        const vector<string> classes=yolact_net.getClassNames();
        for(size_t i=0;i<classes.size();++i)
        {
            cout<<classes[i]<<endl;
        }
    }
    else
    {
        cout<<"Main: yolact Failed. Error message:"<<endl;
        cout<<yolact_net.getErrorDescriptionString()<<endl;
    }
    

    return 0;
}

