#!/usr/bin/python3
#!-*-coding:UTF8-*-

import sys

import cv2 


def demo():
    im=cv2.imread("../img/test_pic.jpg")
    if im is None:
        print("image is empty!")
        return
    cv2.imshow("img",im)
    cv2.waitKey(0)


def disp_img(img):
    cv2.imshow("img from python",img)
    cv2.waitKey(0)
    





