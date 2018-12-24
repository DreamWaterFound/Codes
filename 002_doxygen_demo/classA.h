/**
 * 这里是文件的总体描述？
 * @file classA.h
 * @author Guoqing Liu  (1337841346@qq.com)
 * @brief A类的声明
 * @version 0.1
 * @date 2018-12-24
 * 
 * @copyright WTFPL V2
 * 
 *---------------------------
 * 
 */
#include <iostream>

/**
 * @brief 圆周率的定义
 * 
 */
#define M_PI 3.14159264

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

public:
    //为了说明doxygen功能而加入的：
    /**
    * @brief 打开文件 \n
    * 文件打开成功后，必须使用::CloseFile函数关闭
    * @param[in] fileName    文件名
    * @param[in] fileMode    文件模式，可以由以下几个模块组合而成：\n
    *     -r 读取\n
    *     -w 可写\n
    *     -a 添加\n
    *     -t 文本模式(不能与b联用)\n
    *     -b 二进制模式(不能与t联用)\n
    * @return 返回文件编号\n
    *  --1表示打开文件失败(生成时:.-1)
    * @note文件打开成功后，必须使用::CloseFile函数关闭
    * @par 示例(这里放置par文本w):
    * @code
    *        //用文本只读方式打开文件
    *        int ret = OpenFile("test.txt", "a");
    * @endcode
    * @see A::CloseFile 
    * @deprecated 由于特殊的原因，这个函数可能会在将来的版本中取消
    * @warning 这里是警告信息
    * @remarks 这里是备注信息
    * @todo 这里是将来需要完成的工作
    * @bug 这里是遇到的BUG
    * 
    */
    int OpenFile(const char* fileName, const char* fileMode)
    {
        return 0;
    }
    
     /**
    * @brief 关闭文件
    * @param [in] file    文件
    *
    * @retval 0        成功
    * @retval -1    失败
    * @pre file 必须使用OpenFile的返回值
    */
    int CloseFile(int file)
    {
        return 0;
    }

    /**
     * @brief 获取人物信息
     * @param[in] p 只能输入以下参数：
     * -# a:代表张三        // 生成 1. a:代表张三
     * -# b:代表李四        // 生成 2. b:代表李四
     * -# c:代表王二        // 生成 3. c:代表王二
    */
    void GetPerson(int p)
    {
        p++;
    }

private:
    
    int mnNum;      ///< 对象的值
    int m_variable_1; ///< 成员变量m_variable_1说明
    int m_variable_2; ///< 成员变量m_variable_1说明
    
    /**
     * @brief 成员变量m_c简要说明
     *
     * 成员变量m_variable_3的详细说明，这里可以对变量进行
     * 详细的说明和描述，具体方法和函数的标注是一样的
     */
    bool m_variable_3;
};

/**
 * @brief 命名空间的简单叙述\n
 * 命名空间的详细叙述
 */
namespace sp
{
    /**
     * @brief 用于测试的结构体
     * 
     */
    typedef struct STT_T
    {
        int m1; ///<成员1
        int m2; ///<成员2
    }STT;

    /**
    * @defgroup Group1 title1
    * @{
    */
    int a;
    int b;
    /** @} */

    /**
    * @defgroup Group2 title2
    * @{
    */
    int a2;
    int b2;
    /** @} */

    


    /**
    * @name 组1：PI常量
    * @{
    */
   
    #define PI 3.1415926737 ///< 组1中的圆周率常量
    /** @} */
    
    /**
    * @name 组2：数组固定长度常量
    * @{
    */
   ///数组固定长度常量 in 组2
    const int g_ARRAY_MAX = 1024;
    /** @} */



} // sp
