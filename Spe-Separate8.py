# coding=utf-8
# https://www.cnblogs.com/zhoujinyi/archive/2013/05/07/3064785.html
# txt文件转换为excell文件

import sys
import xlwt #需要的模块


def txt2xls(filename):  #文本转换成xls的函数，filename 表示一个要被转换的txt文本，xlsname 表示转换后的文件名
    print 'converting xls ... '

    xlsname = filename
    f = open(filename+'.txt')   #打开txt文本进行读取
    x = 0                #在excel开始写的位置（y）
    y = 0                #在excel开始写的位置（x）
    xls=xlwt.Workbook()
    # line = f.readline()  # 一行一行读取
    # print line,
    # num = len(line)      # 统计行数
    # print num
    sheet1 = xls.add_sheet('sheet1', cell_overwrite_ok=True) #生成excel的方法，声明excel
    sheet2 = xls.add_sheet('sheet2', cell_overwrite_ok=True)
    sheet3 = xls.add_sheet('sheet3', cell_overwrite_ok=True)
    while True:  #循环，读取文本里面的所有内容
        line = f.readline()   #一行一行读取
        # print line,
        if not line:  #如果没有内容，则退出循环
            break
        for i in line.split('\t'):    #读取出相应的内容写到x #"\t":跳格（移至下一列）
            item=i.strip().decode('utf8')
            sheet1.write(x,y,item)
            y += 1 #另起一列
        x += 1 #另起一行
        y = 0  #初始成第一列

    f.close()
    xls.save(xlsname+'.xls')  #保存
    print ("saving xls successfully")

if __name__ == "__main__":     #当前模块运行时，才运行以下代码
     filename = './语音图片/TIMIT-sa1-white-FFT'
     txt2xls(filename)
     print sys.argv[0]


# #方法二：
# def to_excel(fpath):
#     # 读取参数路径文件
#     f=file(fpath,'r')
#     line = f.read()
#     # 创建workbook
#     w = xlwt.Workbook()
#     # 增加一个sheet页'Sheet1'
#     ws = w.add_sheet('Sheet1')
#     # 以'*'分割，获取每行数据
#     arr_line = line[1:].split('*')
#     for i in range(len(arr_line)):
#         # 对行数据进行遍历，获取行数据元素元组
#         arr_cell = arr_line[i].split('\t')
#         for j in range(len(arr_cell) - 1):
#             # 写入数据
#             ws.write(i, j, arr_cell[j])
#             print '写入(%i,%i):%s' % (i, j, arr_cell[j])
#     fpath_excel=fpath.replace('txt','xls')
#     w.save(fpath_excel)

# 创建txt文件
# f = file('./语音图片/cc.txt', 'w')
# for i in range(1, 6):
#     # 写入数据，每行数据以'*'开头，以'\n'结束，数据行内以制表符'\t'分隔
#     txt = '*a%i\tb%i\tc%i\td%i\te%d\t\n' % (i, i, i, i, i)
#     f.write(txt)
# f.close()

to_excel('./语音图片/dd.txt')

# f = open('./语音图片/aa.txt')    # 返回一个文件对象
# line = f.readline()             # 调用文件的 readline()方法
# while line:
#     print line,                 # 后面跟 ',' 将忽略换行符
#     line = f.readline()
#
# f.close()
#或者
# i = 0
# for line in open ('./语音图片/dd.txt'):
#     print line,
#     a=[]
#     a.append(line)
#     i=i+1
#     print "i:",a

