#include <iostream>
#include <Samples.hpp>
#include <string>
#include <sstream>
#include "K_means.hpp"


using namespace std;

int main(int argc, char* argv[])
{
    cout<<"K-means Test."<<endl;

    if(argc!=4)
    {
        cout<<"Usage: "<<argv[0]<<" samples_path class_num iter_max"<<endl;
        cout<<"Ex: "<<argv[0]<<" ./data/my.txt 2 20"<<endl;
        return 0;
    }

    size_t K,N;

    stringstream ss(argv[2]);
    ss>>K;
    ss=stringstream(argv[3]);
    ss>>N;

    
    Samples sample;
    vector<SampleType> samples;
    samples=sample.getSamples(string(argv[1]));
    if(samples.size()==0)
    {
        cout<<"No samples loaded. Check the file "<<argv[1]<<endl;
        return 0;
    }
    else
    {
        cout<<"File "<<argv[1]<<" loaded complete, "<<samples.size()<<" sample points."<<endl;
    }

    K_MeansCluster cluster(K,N,sample.getRangeHeight(),sample.getRangeWidth(),  2,samples);
    if(cluster.isFailed())
    {
        cout<<"Cluster Failed."<<endl;
        return 0;
    }
   


    


    

    
    

    

    



    return 0;
}