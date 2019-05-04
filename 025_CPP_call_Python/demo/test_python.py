# test_python.py
def hello(s):
    print("python: hello: %s " %s)

def add(a,b):
    print("python: a=%d,b=%d" %(a,b))
    return a+b

def add2(s1,s2):
    print("python: s1=%s,s2=%s" %(s1,s2))
    return s1+s2

class ctest: 
    def __init__(self): 
        print("python: test class nit") 
    def say_hello(): 
        print ("python: test class say hello")
    def say(self, name): 
        print ("python: test class hello: %s" % name) 
        return name
        
hello('zwh')
add(2,4)
add('aaa','bbb')
test = ctest()
test.say('zwt')
