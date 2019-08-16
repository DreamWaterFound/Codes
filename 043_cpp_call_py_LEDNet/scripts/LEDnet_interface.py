#!/usr/bin/python3
#!-*-coding:UTF8-*-

"""
LEDnet 的接口文件
"""
import torch
import os
import numpy as np

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
# 可以考虑留下一个计算预测时间的窗口
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from train.lednet import Net

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["GIO_EXTRA_MODULES"]="/usr/lib/x86_64-linux-gnu/gio/modules/"


def say_hello():
    print("[Python] OK!")

class LEDNet_Interface:
    def __init__(self, weight_path, num_class):
        """
        初始化LEDnet
        """
        self._weight_path=weight_path
        self._model=Net(num_class)
        self._model=torch.nn.DataParallel(self._model)
        self._model=self._model.cuda()
    
        def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            return model

        self._model=load_my_state_dict(self._model, torch.load(weight_path))
    
        print("Model and weights LOADED successfully")
        self._model.eval()

    def eval_image(self, image):
        """
        对一张输入的图像进行评测
        """
        image_tensor=ToTensor()(cv2.resize(np.array(image),(1024,512),cv2.INTER_NEAREST))
        with torch.no_grad():
            pre_origin_result = self._model(Variable(image_tensor.unsqueeze(0).cuda())) 
        
        # 置信度归一化
        pre_unit=torch.nn.functional.softmax(pre_origin_result[0],dim=0)
        # 确定每个像素最可能的类别
        pre=pre_unit.max(0)

        # GPU -> CPU
        # 置信度值
        confidence_image=pre[0].cpu().numpy()
        # 分类标签值
        label_image=pre[1].byte().cpu().numpy()

        return confidence_image, label_image


# 下面的内容是给C++程序的接口，之所以有这个接口是因为目前还没有找到C++构造类的实例时，使用有参数的构造函数的办法
def LEDNet_init(weight_path,num_class):
    global model_interface
    model_interface=LEDNet_Interface(weight_path,num_class)
    return

def LEDNet_eval(image):
    return  model_interface.eval_image(image)
    # print(cimg.dtype)
    # print(limg.dtype)
    # cv2.imshow("origin img",image)
    # cv2.imshow("confindence", cimg)
    # cv2.imshow("label", limg*20)

    # cv2.waitKey(0)
    # return cimg, limg


if __name__ == '__main__':

    LEDNet_init('./models/model_best.pth',20)

    # LEDNet = LEDNet_Interface('./models/model_best.pth',20)

    datadir = '/home/guoqing/Datasets/KITTI/sequences/00/image_2/000000.png'

    img = cv2.imread(datadir,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    c_img, l_img = LEDNet_eval(img)

    # c_img, l_img = LEDNet.eval_image(img)

    cv2.imshow("origin img",img)
    cv2.imshow("confindence", c_img)
    cv2.imshow("label", l_img*20)

    cv2.waitKey(0)
