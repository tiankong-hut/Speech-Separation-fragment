#coding=UTF-8
# http://blog.csdn.net/junli_chen/article/details/53666309
# https://www.cnblogs.com/lhj588/archive/2012/01/06/2314181.html
# http://www.jb51.net/article/60510.htm
# 将数据保存为xls格式--使用xlwt
# xls文件读写--使用xlrd

import numpy as np
import matplotlib.pyplot as plt
import xlwt
import xlrd

#数组保存为xls格式
book=xlwt.Workbook(encoding='utf-8')
sheet=book.add_sheet('Sheet_1', cell_overwrite_ok=True)
data = [[1,2],[3,4],[5,6]]
# for i,value in enumerate(data):
#     print type(value)
    # for j,k in enumerate(value):
    #     print "j,k:",type(k)
        # sheet.write(i,j,k)
# book.save('./语音图片/array1.xls')

'''将数据保存为xls格式--使用xlwt'''''
# import xlwt
# #使用workbook方法，创建一个新的工作簿book
# book=xlwt.Workbook(encoding='utf-8')
# #添加一个sheet
# #当要添加多个sheet时,需要将类似的代码写在后面
# sheet=book.add_sheet('Sheet_1', cell_overwrite_ok=True)
# DATA=(('学号','姓名','年龄','性别','成绩'),
#       ('1001','A','11','男','12'),
#       ('1002','B','12','女','22'),
#       ('1003','C','13','女','32'),
#       ('1004','D','14','男','52'),
#       )
# for i,row in enumerate(DATA):
    # print "i,row:",i,row
    # for j,col in enumerate(row):
        # sheet.write(i,j,col)
        # print "j,col:",type(col)
# book.save('./语音图片/grade.xls')   #如果下面还有sheet要保存，这里可以不要

#创建第二个sheet
# sheet=book.add_sheet('Sheet_2', cell_overwrite_ok=True)
# for i,row in enumerate(DATA):
#     for j,col in enumerate(row):
#         sheet.write(i,j,col)
# book.save('./语音图片/grade.xls')


'''xls文件读写--使用xlrd'''
# data = xlrd.open_workbook('./语音图片/2016-2017学年校历.xls')
# data = xlrd.open_workbook('./语音图片/TIMIT-sa1-white文件/TIMIT-sa1-white-abs.xls')
data = xlrd.open_workbook('./语音图片/THCHS-A2_0_white文件/THCHS-A2_0_white-abs.xls')
print "data:", data

table = data.sheets()[0]   # 通过索引顺序获取，选择不同的工作表,第一个sheet。
print "table:", table

# 获取整行和整列的值（数组）
print "row_values(1):",   table.row_values(2)[0]   #第三行第一列
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
cel=float(cell_A1) + float(cell_A2)
print "cel：",cel

# 获取行数和列数
nrows = table.nrows
ncols = table.ncols
print "nrows, ncols:",nrows, ncols





# lines = []
# with  xlrd.open_workbook('./语音图片/aa.xls') as infile:
#     for line in infile:
#         line = line.rstrip('\n')[1:-1]  # this removes first and last parentheses from the line
#         lines.append([float(v) for v in line.split(',')])
# data=lines

# # 当数字为字符格式时候用
# lines=[]
# for line in cell_A1:
#     # line = line.rstrip('\n')[1:-1]  # this removes first and last parentheses from the line
#     line = line.lstrip('(')   #删除左边的字符
#     line = line.rstrip(')')   #删除右边的字符
#     print "line:",line
#     # lines.append([float(v) for v in line.split(',')])
#     lines.append(v for v in line.split(','))
# cell_A1=lines

'''读取Excel文件'''
data1 = xlrd.open_workbook('./语音图片1/TIMIT-sa1.xls')
data2 = xlrd.open_workbook('./语音图片1/TIMIT-sa1-syn.xls')
print "data:", data1
table1 = data1.sheets()[0]    #选择工作表sheet1
table2 = data2.sheets()[0]
print "table:", table1
'''计算行列数'''
rows_num = table1.nrows
cows_num = table1.ncols
print "行数列数：",rows_num, cows_num
A1 = table1.row_values(0)
A2 = table2.row_values(0)
batch_xs = table1
batch_ys = table2
print "batch_xs:", table1
print "batch_xs:", np.array(A1)
'''打印输出'''
j = 1
for i in A1:
    print "%d:" % j ,i
    j=j+1

