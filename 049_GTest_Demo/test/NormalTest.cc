/**
 * @file NormalTest.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 测试文件
 * @version 0.1
 * @date 2019-10-28
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include <iostream>
#include <Add.h>
#include<gtest/gtest.h>


using std::cout;
using std::endl;

TEST(testCase,test0){
    EXPECT_EQ(Add(2,3),5);
}
int main(int argc,char **argv){
  testing::InitGoogleTest(&argc,argv);
  return RUN_ALL_TESTS();
}

