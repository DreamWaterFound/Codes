#include <iostream>
#include <cstring>

using namespace std;

int main(int argc, char* argv[])
{
    cout<<"A test for core-dump debug on Linux OS."<<endl;
    cout<<"Complied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    size_t c;

    cout<<endl<<"Choose one situation (1-3) :";
    cin>>c;

    int *ptr=nullptr;
    char *strPtr = "test";


    switch(c)
    {
        case 1:
            *ptr=0;
            break;
        case 2:
            ptr=(int*)0;
            *ptr=100;
            break;
        case 3:
            strcpy(strPtr, "TEST");
            break;
        // case 4:
        //     main(0,nullptr);
        //     break;
        default:
            cout<<"Wrong choice."<<endl;
    }

    return 0;
}