#!/usr/bin/python3
#!-*-coding:UTF8-*

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2

im=cv2.imread("../test_pic.jpg")
if im is None:
    print("img is empty.")
else:
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    print("OK.")
# im=im/255.0

# matplotlib.pyplot.imshow(im2)



print("Done.")