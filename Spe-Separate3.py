# coding=utf-8
import wave
# import wavio
import numpy as np
import matplotlib.pyplot as plt
# import  pyaudio

# https://www.cnblogs.com/xingshansi/p/6799994.html
""" 1,批量读取.wav文件名： """
import os
filepath = "./"  # 添加路径,'./'表示当前文件夹
# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
filename = os.listdir(filepath)  # 得到文件夹下的所有文件名称
# 输出所有文件和文件夹
# for file in filename:
#     print(filepath + file)
# f = wave.open(filepath+filename[0],'rb')   #读取一个声音文件
# print "file:",file

""" 2、读取.wav文件 """
# wave.open(file, mode)
# mode可以是：‘rb’，读取文件； ‘wb’，写入文件; 不支持同时读 / 写操作。
# Wave_read.getparams用法：
f = wave.open('compose1.wav', 'rb')   #单通道
params = f.getparams()     #最多6种参数，parameter-参数
nchannels, sampwidth, framerate, nframes = params[:4]
# print nchannels, sampwidth, framerate, nframes
# nchannels:声道数
# sampwidth:量化位数（byte）8,16,32byte
# framerate:采样频率
# nframes:采样点数
strData = f.readframes(nframes)#读取音频，字符串格式
# print len(strData)
waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
# print waveData.shape
# print (max(abs(waveData)))
waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
# waveData = np.reshape(waveData,[nframes,nchannels])
# print waveData[9990:10000]
# plot the wave
time = np.arange(0,nframes)*(1.0 / framerate)   #乘以1.5倍
# print time.shape
# print waveData.shape
# plt.plot(time,waveData)
# plt.xlabel("Time(s)")
# plt.ylabel("Amplitude")
# plt.title("Single channel wavedata")
# plt.grid('on') #标尺，on：有，off:无
# plt.show()


# 多通道：这里通道数为3，主要借助np.reshape一下，其他同单通道处理完全一致
f = wave.open('陌上花开.wav', 'rb')   #双通道
params = f.getparams()     #最多6种参数，parameter-参数
nchannels, sampwidth, framerate, nframes = params[:4]
# print nchannels, sampwidth, framerate, nframes

strData = f.readframes(nframes)#读取音频，字符串格式
waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
# print waveData.shape
waveData = np.reshape(waveData,[nframes,nchannels])  #waveData重新排列：nframes行，nchannels列
f.close()
# plot the wave
time = np.arange(0,nframes)*(1.0 / framerate)
# print "time:",time.shape
# print waveData.shape
# plt.figure()
# plt.subplot(5,1,1)
# # plt.plot(time,waveData[:,0])
# plt.xlabel("Time(s)")
# plt.ylabel("Amplitude")
# plt.title("Ch-1 wavedata")
# plt.grid('on')#标尺，on：有，off:无。
# plt.subplot(5,1,3)
# # plt.plot(time,waveData[:,1])
# plt.xlabel("Time(s)")
# plt.ylabel("Amplitude")
# plt.title("Ch-2 wavedata")
# plt.grid('on')#标尺，on：有，off:无。
# # plt.show()

# 单通道为多通道的特例，所以多通道的读取方式对任意通道wav文件都适用。
# 需要注意的是，waveData在reshape之后，与之前的数据结构是不同的。
# 即waveData[0]等价于reshape之前的waveData，但不影响绘图分析，
# 只是在分析频谱时才有必要考虑这一点。

# # 3、wav写入
# # 参数设置：
# nchannels = 1   #单通道为例
# sampwidth = 2
# fs = 8000
# data_size = len(outData)
# framerate = int(fs)
# nframes = data_size
# comptype = "NONE"
# compname = "not compressed"
# outwave.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
# # 待写入wav文件的存储路径及文件名：
# outfile = filepath+'out1.wav'
# outwave = wave.open(outfile, 'wb')#定义存储路径以及文件名
#  # 数据的写入：
# for v in outData:
#         outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))#outData:16位，-32767~32767，注意不要溢出


import wave
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
''''
# 单通道数据写入：
'''''
# wav文件读取
filepath = "./"  # 添加路径
# filename = os.listdir(filepath)  # 得到文件夹下的所有文件名称
# f = wave.open(filepath + filename[1], 'rb')
# params = f.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]

f = wave.open('THCHS-A2_0.wav', 'rb')
# f = wave.open('Timit_sa1.wav', 'rb')   #单声道
print "f:", f
params = f.getparams()     #最多6种参数，parameter-参数
nchannels, sampwidth, framerate, nframes = params[:4]
print "nchannels, sampwidth, framerate, nframes:",\
       nchannels, sampwidth, framerate, nframes


# f1 = wave.open('T_white.wav', 'rb')
# # f = wave.open('Timit_sa1.wav', 'rb')   #单声道
# print "f1:", f1
# params = f1.getparams()     #最多6种参数，parameter-参数
# nchannels, sampwidth, framerate, nframes= params[:4]
# print "nchannels, sampwidth, framerate, nframes:",\
#        nchannels, sampwidth, framerate, nframes


strData = f.readframes(nframes)  # 读取音频，字符串格式
#  np.fromstring(): 将字符串转化为int
waveData = np.fromstring(strData, dtype=np.int16)
# print waveData[0:50]

dd = waveData
print dd[0:50]
print type(dd)
max_data = max(abs(waveData))
print "max_data:", max_data
waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
# print "waveData:", waveData.shape
# print "waveData:", type(waveData)
# print "waveData:", waveData[0:50]

waveData = waveData[0: nframes]
print "waveData.shape:", waveData

f.close()


'''
# wav文件写入
'''
outData = waveData  # 待写入wav的数据，这里仍然取waveData数据
# print "outData:", outData[0:50]

outfile = filepath + 'out2.wav'
outwave = wave.open(outfile, 'wb')  # 定义存储路径以及文件名
nchannels = 1    #单通道
sampwidth = 2
fs = 16000
data_size = len(outData)
framerate = int(fs)
nframes = data_size
comptype = "NONE"
compname = "not compressed"
outwave.setparams((nchannels, sampwidth, framerate, nframes,
                  comptype, compname))


# 数据写入
for v in outData:
    n=1    #n在此处可以调节声音大小
    outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))  # outData:16位，-32767~32767，注意不要溢出
    # outwave.writeframes(struct.pack('h', int(v* max_data* n)))      #pack成为二进制字符串
          # struct.pack用于将Python的值根据格式符，转换为字符串! 'h'表示（short）integer类型，占两个字节
    # for v in dd:
    #     outwave.writeframes(struct.pack('h', int(v) ))
outwave.close()

'''''
将list中的数据写入到excel文件中
'''''
waveData = waveData[0:50000]
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
# # # wav文件读取
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


# ##### 加噪声方法一
# sample_len = 132300    #音频的字节长度
# noise_file = open('noise.wav','w')
# for i in range(0,sample_len):
#     value = np.random.randint(-32767, 32767)
#     packed_vaule = struct.pack('h', value)
#     noise_file.writeframes(packed_vaule)
#
# #####方法二
# noise_file = open('noise.pcm','w')
# for i in range(0,sample_len):
#     value = np.random.randint(-32767, 32767)
#     packed_vaule = struct.pack('h', value)
#     noise_file.write(packed_vaule)

'''''
4、音频播放
wav文件的播放需要用到pyaudio
'''''
import wave
# import pyaudio
import os

# wav文件读取
# filepath = "./data/"  # 添加路径
# filename = os.listdir(filepath)  # 得到文件夹下的所有文件名称
# f = wave.open(filepath + filename[0], 'rb')
# params = f.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]
# # instantiate PyAudio
# p = pyaudio.PyAudio()
# # define stream chunk
# chunk = 1024
# # 打开声音输出流
# stream = p.open(format=p.get_format_from_width(sampwidth),
#                 channels=nchannels,
#                 rate=framerate,
#                 output=True)
#
# # 写声音输出流到声卡进行播放
# data = f.readframes(chunk)
# i = 1
# while True:
#     data = f.readframes(chunk)
#     if data == b'': break
#     stream.write(data)
# f.close()
# # stop stream
# stream.stop_stream()
# stream.close()
# # close PyAudio
# p.terminate()

'''''
5、信号加窗
通常对信号截断、分帧需要加窗，因为截断都有频域能量泄露，而窗函数可以减少截断带来的影响
窗函数在scipy.signal信号处理工具箱中，如hamming窗：
'''''
import pylab as pl
import scipy.signal as signal
# pl.figure(figsize=(6,2))       #figsize表示窗的长宽
# pl.plot(signal.hanning(512))
# plt.show()

'''''
6、信号分帧
信号分帧的理论依据，其中x是语音信号，w是窗函数
加窗截断类似采样，为了保证相邻帧不至于差别过大，通常帧与帧之间有帧移，其实就是插值平滑的作用。
'''''
# np.repeat：主要是直接重复
# np.tile：主要是周期性重复
# 对应分帧的代码实现：
# 这是没有加窗的示例：
import numpy as np
import wave
import os

# import math

def enframe(signal, nw, inc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    signal_length = len(signal)  # 信号总长度
    if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))
    pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                           (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    #    win=np.tile(winfunc(nw),(nf,1))  #window窗函数，这里默认取1
    #    return frames*win   #返回帧信号矩阵
    return frames


def wavread(filename):
    f = wave.open(filename, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    f.close()
    waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
    waveData = np.reshape(waveData, [nframes, nchannels]).T
    return waveData

# filepath = "./data/"  # 添加路径
# dirname = os.listdir(filepath)  # 得到文件夹下的所有文件名称
# filename = filepath + dirname[0]
# f = wave.open('陌上花开.wav', 'rb')   #双通道
filename= '陌上花开.wav'
data = wavread(filename)
nw = 512
inc = 128
Frame = enframe(data[0], nw, inc)


# 如果需要加窗，只需要将函数修改为：
def enframe(signal, nw, inc, winfunc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    signal_length=len(signal) #信号总长度
    if signal_length<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf=int(np.ceil((1.0*signal_length-nw+inc)/inc))

    pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal=np.concatenate((signal,zeros)) #填补后的信号记为pad_signal
    indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    win=np.tile(winfunc,(nf,1))  #window窗函数，这里默认取1
    return frames*win   #返回帧信号矩阵
# 其中窗函数，以hamming窗为例：
# winfunc = signal.hamming(nw)
# Frame = enframe(data[0], nw, inc, winfunc)



''''
7、语谱图
其实得到了分帧信号，频域变换取幅值，就可以得到语谱图，如果仅仅是观察，
matplotlib.pyplot有specgram指令：
'''''
import wave
import matplotlib.pyplot as plt
import numpy as np
import os

filepath = "./"  # 添加路径
# filename = os.listdir(filepath)  # 得到文件夹下的所有文件名称
# f = wave.open(filepath + filename[0], 'rb')
# params = f.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]
f = wave.open('陌上花开.wav', 'rb')   #双通道
params = f.getparams()     #最多6种参数，parameter-参数
nchannels, sampwidth, framerate, nframes = params[:4]
print nchannels, sampwidth, framerate, nframes
strData = f.readframes(nframes)  # 读取音频，字符串格式
waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
waveData = np.reshape(waveData, [nframes, nchannels]).T    #转置
f.close()
# plot the wave
# plt.specgram(waveData[0], Fs=framerate, scale_by_freq=True, sides='default')
# # plt.ylim(0,5000)
# plt.ylabel('Frequency(Hz)')
# plt.xlabel('Time(s)')
# plt.show()
'''
specgram:功率谱  ，同spectrogram？
Time-dependent frequency analysis (spectrogram)
功率谱是功率谱密度函数的简称，它定义为单位频带内的信号功率。它表示了信号功率随着频率的变化情况，
即信号功率在频域的分布状况。功率谱表示了信号功率随着频率的变化关系。常用于功率信号（区别于能量信号）
的表述与分析，其曲线（即功率谱曲线）一般横坐标为频率，纵坐标为功率。周期性连续信号x(t)的频谱可表示为
离散的非周期序列Xn,它的幅度频谱的平方│Xn│2所排成的序列，就被称之为该周期信号的“功率谱”。
傅立叶级数提出后，首先在人们观测自然界中的周期现象时得到应用。19世纪末，Schuster提出用傅立叶级数
的幅度平方作为函数中功率的度量，并将其命名为“周期图”（periodogram）。这是经典谱估计的最早提法，
这种提法至今仍然被沿用，只不过现在是用快速傅立叶变换（FFT）来计算离散傅立叶变换（DFT），
用DFT的幅度平方作为信号中功率的度量。
'''


'''
本篇尝试使用Python对音频文件进行频谱分析。
在语音识别领域对音频文件进行频谱分析是一项基本的数据处理过程，同时也为后续的特征分析准备数据。
'''
# http://blog.sina.com.cn/s/blog_40793e970102w3m2.html

import wave
# import pyaudio       #未安装好
import numpy
import pylab

# #打开WAV文档，文件路径根据需要做修改
# # wf = wave.open("陌上花开.wav", "rb")
# wf = wave.open('陌上花开.wav', 'rb')
# #创建PyAudio对象
# p = pyaudio.PyAudio()
# stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
# channels=wf.getnchannels(),
# rate=wf.getframerate(),
# output=True)
# nframes = wf.getnframes()
# framerate = wf.getframerate()
# #读取完整的帧数据到str_data中，这是一个string类型的数据
# str_data = wf.readframes(nframes)
# wf.close()
# #将波形数据转换为数组
# # A new 1-D array initialized from raw binary or text data in a string.
# wave_data = numpy.fromstring(str_data, dtype=numpy.short)
# #将wave_data数组改为2列，行数自动匹配。在修改shape的属性时，需使得数组的总长度不变。
# wave_data.shape = -1,2
# #将数组转置
# wave_data = wave_data.T
# #time 也是一个数组，与wave_data[0]或wave_data[1]配对形成系列点坐标
# time = numpy.arange(0,nframes)*(1.0/framerate)
# # 绘制波形图
# pylab.plot(time, wave_data[0])
# pylab.subplot(212)
# pylab.plot(time, wave_data[1], c="g")
# pylab.xlabel("time (seconds)")
# pylab.show()
