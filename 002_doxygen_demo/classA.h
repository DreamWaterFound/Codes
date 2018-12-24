#include <iostream>

using namespace std;

class A{
public:
    A();
    A(int n);
    ~A();

public:
    void setNum(int n);
    int getNum(void);

private:
    int mnNum;
};