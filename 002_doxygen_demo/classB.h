#include <iostream>

using namespace std;

class B{
public:
    B();
    B(int n);
    ~B();

public:
    void setNum(int n);
    int getNum(void);

private:
    int mnNum;
};