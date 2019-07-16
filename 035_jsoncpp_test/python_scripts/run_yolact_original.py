#!/usr/bin/python3
#!-*-coding:UTF8-*-

"""
冯永晖给的把YOLACT预测结果存储的程序demo
"""

# 导入必要依赖

# added by guoqing, to solve cv2 problem
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from yolact import Yolact
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess
import pycocotools
from data import cfg, set_cfg

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import json
import os

import cv2


class Detections(object):
    """
    保存预测结果的类
    """

    def __init__(self, output_dir, output_name="detection"):
        """
        构造函数，参数：
            - output_dir            输出文件夹
            - output_name           输出文件的名称
        构造函数就干一件事，设置输出路径，如果不存在就创建
        """
        # 存储实例数据的字典,其键为后文中的image_name
        self.instance_data = {}
        # 将多个路径结合后返回
        self.output_dir = os.path.join(output_dir, output_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    
    def add_instance(self, image_name: str, idx: int, category_id: int, bbox: list, segmentation: np.ndarray, score: float):
        """ 
            向该类中添加一个检测出来的实例
            参数：
                - image_name            图像名称 # ? 啥名称啊
                - idx                   类别id  # ? 看上去并不是这样
                - category_id           类别id
                - bbox                  bounding box,不过注意这里是以列表的方式存储的
                - segmentation          分割，也就是mask，numpy类型的
                - score                 该物体实例的评分
            
            Note that bbox should be a list or tuple of (x1, y1, x2, y2) 
        """
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        # ? 为什么这里进行四舍五入就能够避免较大的文件体积?
        bbox = [round(float(x) * 10) / 10 for x in bbox]
        # encode segmentation with RLE
        # segmentation.astype => 转换数据类型
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))  
        # NOTE 不太明白这个是做啥的,暂时不追究
        rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings
        # create one instance json
        # 返回字典中
        dict_instance = self.instance_data.get(image_name, {})
        # 感觉json中的花括号对应Python中的元组,键值对也是和python中对应的,所以下面的代码看上去就是和真正的json文件写法一样
        dict_instance.update(
            {
                idx:
                {
                    'category_id': int(category_id),  # TODO: origin: get_coco_cat(int(category_id))
                    'bbox': bbox,
                    "segmentation": rle,
                    'score': float(score)
                }
            }
        )
        # 父项?
        self.instance_data.update({image_name: dict_instance})

    def dump_all(self):
        """
            将已经有的数据都保存到json文件中
        """
        for image_name, dict_instance in self.instance_data.items():
            # load .json file
            # with open(os.path.join(self.output_dir, image_name + ".json"), "w") as f:
            #     f.write(json.dumps(dict_instance))
            # load .bin file
            # 竟然是以二进制的方式读写诶
            with open(os.path.join(self.output_dir, image_name + ".bin"), "wb") as f:
                # *.encode() ==> 将Python对象编码成字符串
                f.write(json.dumps(dict_instance).encode())


class RunYolact(object):
    """
    运行YOLACT的类
    source: https://github.com/dbolya/yolact/issues/9
    """
    def __init__(self, trained_model:str, save_json=True, output_dir=None, output_name="detection", output_num=5):
        """
        YOLACT 初始化,参数:
            - save_json         是否将计算结果保存为json文件
            - output_dir        当上个参数为True时,这个参数表示将json文件保存到的位置
            - output_name       保存的json文件名
            - output_num        # ? 目测是要输出的类别个数
        """
        #  step 0 初始化变量
        self.save_json = save_json
        # NOTE 卧槽还有这种用法,学习了
        self.detections = None
        self.output_num = output_num
        # step 1 如果指定了要生成json文件,就创建上面的Detection类对象
        if self.save_json and output_dir is not None:
            self.detections = Detections(output_dir, output_name)
        # step 2 初始化YOLACT网络
        with torch.no_grad():
            set_cfg("yolact_base_config")
            torch.cuda.set_device(1)
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.net = Yolact()
            # TODO 这里的权值是需要进行修改的
            # self.net.load_weights('./weights/yolact_base_54_800000.pth')
            self.net.load_weights(trained_model)
            self.net.eval()
            self.net = self.net.cuda()
        print("load model complete")

    def run_once(self, src, image_name):
        """
        只对一张图像进行预测.参数:
            - src           # ? 要预测的图像
            - image_name    图像名称 # ? 猜测就是图像的文件名
        """
        # step 0 准备
        self.net.detect.cross_class_nms = True
        self.net.detect.use_fast_nms = True
        cfg.mask_proto_debug = False
        # step 1 预测
        with torch.no_grad():
            frame = torch.Tensor(src).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            time_start = time.clock()
            preds = self.net(batch)
            time_elapsed = (time.clock() - time_start)
            h, w, _ = src.shape
            # NOTICE 这里并没有设置最小的阈值
            t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=0.)  # TODO: give a suitable threshold
            torch.cuda.synchronize()
            classes, scores, boxes, masks = [x[:self.output_num].cpu().numpy() for x in t]  # TODO: Only 5 objects for test
            print(time_elapsed)
            # 将预测得到的每一个结果都添加到detection对象中
            for i in range(masks.shape[0]):
                self.detections.add_instance(image_name, i, classes[i], boxes[i, :], masks[i, :, :], scores[i])
        # step 2 保存所有预测结果
        self.detections.dump_all()



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    import glob
    import argparse

    # step 1 参数解析
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation and save result in json.')

    parser.add_argument('--trained_model',
                        default='../pre_models/yolact_darknet53_54_800000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')

    parser.add_argument('--save_json',
                        default=False, type=str2bool,
                        help='Enable or disable generate json files with save the results of prediction.')

    parser.add_argument('--seq_path',
                        type=str,
                        help='The image sequences path.')

    parser.add_argument('--output_dir',
                        type=str,
                        help="The json file's output path.")

    parser.add_argument('--output_name',
                        type=str,
                        help="The json file's name.")

    parser.add_argument('--output_num',
                        default=5, type=int,
                        help="The json file's name.")

    args = parser.parse_args(argv)

    # step 2 生成模型
    model = RunYolact(save_json=args.save_json, output_dir=args.output_dir, output_name=args.output_name, output_num=args.output_num,trained_model=args.trained_model)

    # step 3 预测
    li_src = glob.glob(os.path.join(args.seq_path, "/*.png"))
    for src_name in li_src:
        # image_name 其实就是文件名
        image_name = os.path.basename(src_name)[:-4]
        src = cv2.imread(src_name)
        model.run_once(src, image_name)

    







    # # args
    # seq_name = "rgbd_dataset_freiburg1_desk"
    # model_name = "ResNet101-FPN"
    # output_dir = os.path.join("./results/TUM/", seq_name, model_name)
    # output_name = "detections-bin"
    # output_num = 5

    # li_src = glob.glob(os.path.join("/home/fyh/Datasets/TUM/", seq_name, "rgb/*.png"))
    # model = RunYolact(save_json=True, output_dir=output_dir, output_name=output_name, output_num=output_num)
    # for src_name in li_src:
    #     # image_name 其实就是文件名
    #     image_name = os.path.basename(src_name)[:-4]
    #     src = cv2.imread(src_name)
    #     model.run_once(src, image_name)