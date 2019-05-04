#!/usr/bin/python3
#!-*-coding:UTF8-*-

# 一个单独的无参数函数
def say_hello():
    """无参数的函数"""
    print("Python: Hello world!")

# 字符串为参数的函数
def greet_user(user_name):
    print("Python: Welcome to py world, "+user_name.title()+" !")

# 数字作为参数x2的函数
def add_2_numbers(num1,num2):
    sum=num1+num2
    print("sum of "+str(num1)+" and "+str(num2)+" is "+str(sum)+".")

# 数字作为参数，并且具有数值返回值的函数
def multi_2_numbers(num1,num2):
    product=num1*num2
    return product

# 返回值为字符串类型的函数
def get_user_name():
    return 'Guoqing'

# 定义一个类
class Student():
    """学生类"""

    def __init__(self,name="default",id=0,score=0):
        self.name=name
        self.id=id
        self.score=score
        print("Python: Student Instance created.")

    def increase_score(self,d_score):
        self.score+=d_score
        print("Python: score = "+str(d_score))

    def greet(self):
        print("Python: Hello, "+self.name+"!")

    def say(self,str):
        print("Python: Hello, "+str+"!")
        return str;

class ctest: 
    def __init__(self): 
        print("python: test class nit") 
    def say_hello(self): 
        print ("python: test class say hello")
    def say(self, name): 
        print ("python: test class hello: %s" % name) 
        return name

    








