#include <iostream>
#include <string>
#include <sstream>

#include "include/types/type_2D_point.hpp"
#include "include/Cluster/K_means_2D_point.hpp"
#include "include/tools/SampleSets_2D_point.hpp"

using namespace cv;
using namespace std;


int main(int argc, char* argv[])
{
    cout<<"K-means on 2D points Test."<<endl;

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

    cout<<"K="<<K<<endl;



    Samples2D sets;
    vector<Points_2D> vpts;
    vpts=sets.getSamples(string(argv[1]));
    if(vpts.size()==0)
    {
        cout<<"No samples loaded. Check the file "<<argv[1]<<endl;
        return 0;
    }
    else
    {
        cout<<"File "<<argv[1]<<" loaded complete, "<<vpts.size()<<" sample points."<<endl;
    }

    K_MeansCluster2DPoint cluster(K,N,0.1,sets.getRangeHeight(),sets.getRangeWidth(),vpts);
    cluster.Compute();
    if(cluster.isFailed())
    {
        cout<<"Cluster Failed."<<endl;
        return 0;
    }
    else
    {
        cout<<"Cluster Completed."<<endl;
        return 0;
    }
    
   


}