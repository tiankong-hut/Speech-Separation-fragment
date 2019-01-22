# coding=utf-8
# https://www.cnblogs.com/jackchiang/p/4605327.html
# https://www.cnblogs.com/sunny3312/archive/2017/01/07/6260472.html
import os
import numpy as np

# 按时间存取先后列出
DIR = "./语音图片"
print "os.stat:",os.stat("./语音图片")      #属性
def compare(x, y):    #按照时间顺序
    stat_x = os.stat(DIR + "/" + x)
    stat_y = os.stat(DIR + "/" + y)
    if stat_x.st_ctime < stat_y.st_ctime:   #文件创建时间,小的在前面创建
        return -1
    elif stat_x.st_ctime > stat_y.st_ctime:
        return 1
    else:
        return 0

iterms = os.listdir(DIR)
iterms.sort(compare)    #按照文件创建时间先后顺序排序
# cmp是自建函数：按字母顺序排序
iterms.sort(cmp)        #cmp是自建函数:，Return negative if x<y, zero if x==y, positive if x>y.
for iterm in iterms:
    print iterm


###
dir = "./语音图片"
list = os.listdir(dir)
print "len(list):",len(list)              #打印文件个数
print "path:", os.path.join(dir,list[1])  #打印某个文件
for i in range(0,len(list)):
      path = os.path.join(dir,list[i])
      # if os.path.isfile(path):
      #     print "path:",path


# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        # print child
        # print child.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题

eachFile("./语音图片")


# 原址排序,列表有自己的sort方法，其对列表进行原址排序
x = [4, 6, 2, 1, 7, 9]
x.sort()
print x # [1, 2, 4, 6, 7, 9]

#副本排序,[:]分片方法,注意：y = x[:] 通过分片操作将列表x的元素全部拷贝给y，
# 如果简单的把x赋值给y：y = x，y和x还是指向同一个列表，并没有产生新的副本。
x =[4, 6, 2, 1, 7, 9]
y = x[ : ]
y.sort()
print x #[4, 6, 2, 1, 7, 9]
print y #[1, 2, 4, 6, 7, 9]
# sorted返回一个有序的副本，并且类型总是列表
y = sorted(x)
print y #[1, 2, 4, 6, 7, 9]
print sorted('Python') #['P', 'h', 'n', 'o', 't', 'y']


# # 读取文件内容并打印
# def readFile(filename):
#     fopen = open(filename, 'r')  # r 代表read
#     for eachLine in fopen:
#         print "读取到得内容如下：", eachLine
#     fopen.close()
#
#
# # 输入多行文字，写入指定文件并保存到指定文件夹
# def writeFile(filename):
#     fopen = open(filename, 'w')
#     print "\r请任意输入多行文字", " ( 输入 .号回车保存)"
#     while True:
#         aLine = raw_input()
#         if aLine != ".":
#             fopen.write('%s%s' % (aLine, os.linesep))
#         else:
#             print "文件已保存!"
#             break
#     fopen.close()
#
#
# if __name__ == '__main__':
#     filePath = "D:\\FileDemo\\Java\\myJava.txt"
#     filePathI = "D:\\FileDemo\\Python\\pt.py"
#     filePathC = "C:\\"
#     eachFile(filePathC)
#     readFile(filePath)
#     writeFile(filePathI)
