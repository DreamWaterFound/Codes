#!/usr/bin/python3
#!-*-coding:UTF8-*-

# 这个文件就是用来处理文件路径问题的

import sys


def init_sys_path():
    if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

