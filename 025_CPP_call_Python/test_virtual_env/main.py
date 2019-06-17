#!/usr/bin/python3
#!-*-coding:UTF8-*-

import sys

import torch
import numpy as np
from pylab import *

import matplotlib


# 一个单独的无参数函数
def say_hello():
    """无参数的函数"""
    
    print("======== Python Function 'say_hello()' =============")
    print("Python: Hello world!")
    print("Loading torch ...")
    print("OK.")
    print(torch.Tensor([[2,3]]))
    print(np.arange(15).reshape(3,5))

    X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
    C,S = np.cos(X), np.sin(X)
    plot(X,C)
    plot(X,S)

    show()

    print("====================================================")

def init_sys_path():
    if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

def show_img():
    import cv2
    im=cv2.imread("../test_pic.jpg")
    if im is None:
        print("image is empty!")
        return
    cv2.imshow("img",im)
    cv2.waitKey(0)
#     im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
#     im2=im/255.0
#     matplotlib.pyplot.imshow(im2)
    






