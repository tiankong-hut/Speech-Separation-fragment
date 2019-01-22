# coding=utf-8
import xlrd


import os
# 按时间存取先后列出
DIR = "./语音图片/训练集"
print "os.stat:",os.stat("./语音图片/训练集")      #属性
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
# iterms.sort(compare)    #按照文件创建时间先后顺序排序
# cmp是自建函数：按字母顺序排序
iterms.sort(cmp)          #cmp是自建函数:，Return negative if x<y, zero if x==y, positive if x>y.
for iterm in iterms:
    print iterm
    iterm = "./语音图片/训练集/" + iterm
    data = xlrd.open_workbook(iterm)
    table = data.sheets()[0]          #选择不同的sheet
    cell_A = table.col_values(0)      #第一列第三行
    print "mm:",cell_A[1]             #第一列第三行


###
data = xlrd.open_workbook('./语音图片/THCHS-A2_0_white文件/THCHS-A2_0_white-abs.xls')
print "data:", data
# 通过索引顺序获取，选择不同的工作表,第一个sheet。
table = data.sheets()[0]
print "table:", table
# 获取整行和整列的值（数组）
print "row_values(2)[0]:",table.row_values(2)[0]      #第三行第一列
print "col_values(0)[2]:",table.col_values(0)[2]   #第一列第三行
# 使用行列索引,得到数据
cell_A1 = table.row(2)[0].value    #第3行第1列
cell_A2 = table.col(0)[2].value    #第1列第3行
# # 单元格,用行列坐标定位
# cell_A1 = table.cell(0, 0).value
# cell_A2 = table.cell(2, 0).value
print "type(cell_A1):",type(cell_A1)
print "cell_A1:",cell_A1
print "cell_A2:",cell_A2
# cel=float(cell_A1) + float(cell_A2)
# print "cel：",cel





###
dir = "./语音图片"
list = os.listdir(dir)
print "len(list):",len(list)                #打印文件个数
print "path:", os.path.join(dir,list[1])    #打印某个文件
for i in range(0,len(list)):
      path = os.path.join(dir,list[i])
      # if os.path.isfile(path):
      #     print "path:",path


# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        # print child
        # print child.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题

# dir = "./语音图片"
# eachFile(dir)

