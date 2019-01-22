# coding=utf-8
import os
# 同一文件目录下:
import  shiyan_2   #调用shiyan_2的函数
from shiyan_2 import compare  #加了这句就不用加模块名的前缀了

DIR = "./语音图片"
iterms = os.listdir(DIR)
iterms.sort(shiyan_2.compare)   #加了“shiyan_2”的前缀
iterms.sort(compare)            #不加“shiyan_2”前缀
# iterms.sort(cmp)                #按字母顺序排列
for ite in iterms:
    print ite


# 不同文件目录下: 在/Neural network/目录下调用shiyan_3
# 若不在同一目录，python查找不到，必须进行查找路径的设置，将模块所在的文件夹加入系统查找路径
import sys
sys.path.append("/home/chuwei/PycharmProjects/Neural network")
import shiyan_3
from shiyan_3 import func
# shiyan_3.func()
# func()

# if __name__ == "__main__":
#     print sys.argv[0]     #当前文件地址
#     print("shiyan_1.py is being run directly")
# else:
#     print("shiyan_1.py is being imported into another module")
