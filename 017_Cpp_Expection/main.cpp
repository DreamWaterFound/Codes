#include <iostream>
#include <exception>
#include <vector>


using namespace std;


int main(int argc, char* argv[])
{

    cout<<"Execption Test."<<endl;

    vector<int> a;
    try
    {
        cout<<a[4]<<endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        cout<<"An error occured."<<endl;
    }
    

    
    



    return 0;
}

