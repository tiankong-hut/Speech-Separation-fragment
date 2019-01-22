# coding=utf-8

from __future__ import division     #必须放在最前面，division进行精确的除法运算
import wave
import matplotlib.pyplot as plt
import numpy as np
import os
import struct

'''''
# 单通道数据写入：
'''''
# wav文件读取
# filename = os.listdir(filepath)  # 得到文件夹下的所有文件名称
# f = wave.open(filepath + filename[1], 'rb')
# params = f.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]

filepath = "./"  # 添加路径
# wav_open = wave.open('THCHS-A2_0.wav', 'rb')
wav_open = wave.open('Timit_sa1.wav', 'rb')
print "wav_open:",  wav_open
params = wav_open.getparams()     #最多6种参数，parameter-参数
nchannels, sampwidth, framerate, nframes = params[:4]
print "nchannels, sampwidth, framerate, nframes:",\
       nchannels, sampwidth, framerate, nframes
len_n = nframes    # 短点的长度
ndata = wav_open.readframes(nframes)  # 读取音频，字符串格式
ndata = np.fromstring(ndata, dtype=np.int16)  #  np.fromstring(): 将字符串转化为int
print "ndata:",ndata
print "ndata_max:", max(abs(ndata))
ndata= ndata * 1.0 / (max(abs(ndata)))  # wave幅值归一化
print "ndata_max_max:", max(abs(ndata))

wav_open_1 = wave.open('T_white.wav', 'rb')
params = wav_open_1 .getparams()
nchannels, sampwidth, framerate, nframes= params[:4]
print "nchannels, sampwidth, framerate, nframes:",\
       nchannels, sampwidth, framerate, nframes
ndata_1 = wav_open_1.readframes(nframes)  # 读取音频，字符串格式
ndata_1 = np.fromstring(ndata_1, dtype=np.int16)
print "ndata_1:",ndata_1
print "ndata_1_max:", max(abs(ndata_1))
ndata_1 = 0.05*ndata_1 * 1.0 / (max(abs(ndata_1)))  # ndata_1幅值归一化
print "ndata_1:",ndata_1


'''合成语音'''
ndata_compose = []
for i in range(0,len_n):
    com_data = ndata[i] + ndata_1[i]
    ndata_compose.append(com_data)
ndata_compose = np.array(ndata_compose)  #list转换为array
print "ndata_compose:",ndata_compose.shape


ndata_compose = ndata_compose  * 1.0 / (max(abs(ndata_compose)))  # 幅值归一化
print "ndata_compose_max:", max(abs(ndata_compose))

ndata_compose = ndata_compose[0: len_n]
print "ndata_compose.shape:", ndata_compose.shape

wav_open.close()
wav_open_1.close()

'''
# wav文件写入
'''
out_data = ndata_compose   # 待写入wav的数据，这里仍然取ndata_compose数据
# print "outData:", outData[0:50]

outfile = filepath + 'out1.wav'
outwave = wave.open(outfile, 'wb')  # 定义存储路径以及文件名
nchannels = 1    #单通道
sampwidth = 2
fs = 16000
data_size = len(out_data)
framerate = int(fs)
nframes = data_size
comptype = "NONE"
compname = "not compressed"
outwave.setparams((nchannels, sampwidth, framerate, nframes,
                  comptype, compname))


# 数据写入
# for v in out_data:
#     n=1    #n在此处可以调节声音大小
#     outwave.writeframes(struct.pack('h', int(v * 64000/2*n)))  # outData:16位，-32767~32767，注意不要溢出
#     # outwave.writeframes(struct.pack('h', int(v* max_data* n)))      #pack成为二进制字符串
#        # struct.pack用于将Python的值根据格式符，转换为字符串! 'h'表示（short）integer类型，占两个字节
# outwave.close()
# print "已生成wav文件"

'''''
将list中的数据写入到excel文件中
'''''
waveData = ndata_compose[0:50000]
# convert list to array
data_array = np.array(waveData)
print "data_array:", data_array.shape
# print "data_array:", type(data_array)
# print "data_array:", data_array[0:50]
# saving...
# np.savetxt('data.xls', data_array, delimiter=',')
print 'Finish saving xls file'








# """
# # 多通道数据写入：
# # 多通道的写入与多通道读取类似，多通道读取是将一维数据reshape为二维，
# # 多通道的写入是将二维的数据reshape为一维，其实就是一个逆向的过程：
# """
# # wav文件读取
# filepath = "./"  # 添加路径,'./'表示当前路径
# f = wave.open('陌上花开.wav', 'rb')   #双通道
# params = f.getparams()     #最多6种参数，parameter-参数
# nchannels, sampwidth, framerate, nframes = params[:4]
# print nchannels, sampwidth, framerate, nframes
# #     2 2 44100 793800
# strData = f.readframes(nframes)  # 读取音频，字符串格式
# waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
#
# max_data = max(abs(waveData))
# print "max_data:",max_data
# waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
# waveData = np.reshape(waveData, [nframes, nchannels])  #转换为二维数据
# f.close()
#
# # wav文件写入
# outData = waveData  # 待写入wav的数据，这里仍然取waveData数据
# outData = np.reshape(outData, [nframes * nchannels, 1])  #转化为一维数据
#
# outfile = filepath + 'out2.wav'
# outwave = wave.open(outfile, 'wb')  # 定义存储路径以及文件名
# nchannels = 2     #双声道
# sampwidth = 2
# fs = 44100
# data_size = len(outData)
# framerate = int(fs)
# nframes = data_size
# comptype = "NONE"
# compname = "not compressed"
# outwave.setparams((nchannels, sampwidth, framerate, nframes,
#                  comptype, compname))
# for v in outData:
#    # outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))  # outData:16位，-32767~32767，注意不要溢出
#    outwave.writeframes(struct.pack('h', v * max_data))   # max_data
# outwave.close()

