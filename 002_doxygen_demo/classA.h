/**
 * @file classA.h
 * @author Guoqing Liu (1337841346@qq.com)
 * @brief A类的声明
 * @version 0.1
 * @date 2018-12-24
 * 
 * @copyright WTFPL V2
 * 
 */
#include <iostream>

using namespace std;

/**
 * @brief 类A，存储一个变量，提供两个参数
 * <p>这里写东西也算数吗<\\p>
 * 
 */
class A{
public:
    /**
     * @brief A类函数的无参数构造函数
     * 
     */
    A();

    /**
     * @brief A类函数的有参数构造函数
     * 
     * @param n 对象的值
     */
    A(int n);

    /**
     * @brief A类对象的析构函数
     * 
     */
    ~A();

public:

    /**
     * @brief 设置对象的数值
     * 
     * @param n 要设置的数值
     */
    void setNum(int n);

    /**
     * @brief 获取对象的数值
     * 
     * @return int 返回对象的数值
     */
    int getNum(void);

private:
    
    int mnNum;      ///< 对象的值
};