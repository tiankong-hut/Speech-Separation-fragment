# coding=utf-8
#定义函数，用于被shiyan_1调用
import os

def compare(x,y):    #按照时间顺序
    DIR = "./语音图片"
    stat_x = os.stat(DIR + "/" + x)
    stat_y = os.stat(DIR + "/" + y)
    if stat_x.st_ctime < stat_y.st_ctime:   #文件创建时间,小的在前面创建
        return -1
    elif stat_x.st_ctime > stat_y.st_ctime:
        return 1
    else:
        return 0

