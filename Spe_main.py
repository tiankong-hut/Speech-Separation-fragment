# coding=utf-8
import os
import sys
import shiyan_2
from shiyan_2 import compare


DIR = "./语音图片/"
iterms = os.listdir(DIR)
iterms.sort(shiyan_2.compare)   #加了“shiyan_2”的前缀
# iterms.sort(compare)            #不加“shiyan_2”前缀
# iterms.sort(cmp)                #按字母顺序排列
for i in iterms:
    print i





if __name__ == "__main__":   #在当前模块运行时，执行以下语句，否则执行else语句。
    print "sys.argv[0] :", sys.argv[0]     #当前文件地址
    print("shiyan_2.py is being run directly")
else:
    print("shiyan_2.py is being imported into another module")