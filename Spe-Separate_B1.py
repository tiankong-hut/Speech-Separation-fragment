# coding=utf-8

from __future__ import division     #必须放在最前面，division进行精确的除法运算
import matplotlib.pyplot as plt
import xlwt
import wavio
import numpy as np
from scipy import signal
import pylab


''''' 声谱图 '''''
# wav_s=wavio.read("Timit_sa1.wav")
wav_s=wavio.read("TIMIT-sa1-white.wav")
print "wav_s:\n",wav_s
# Wav(data.shape=(46000, 1), data.dtype=int16, rate=16000, sampwidth=2)
wav=wav_s.data[:,0].astype(float)/np.power(2,wav_s.sampwidth*8-1)   #astype-变量类型转换，power-幂指数

''''STFT变换'''
# plt.specgram(wav, Fs=16000, NFFT=1024,noverlap=512,scale_by_freq=True, sides='default')
# [f,t,x]=signal.spectral.spectrogram(wav, 16000, np.hamming(1024), nperseg=1024,
#                noverlap=512, detrend=False, return_onesided=True, mode='magnitude') #此处x为实数
[f,t,x]=signal.spectral.stft(wav, 16000, np.hamming(1024), nperseg=1024,
               noverlap=512, detrend=False, return_onesided=True)  #此处x为复数
''''ISTFT变换'''
[T,X] = signal.spectral.istft(x, 16000,  nperseg=1024,       #此处x为复数
               input_onesided=True)
print "ISTFT-X:",X.shape
''''画出ISTFT的时域波形图'''
# pylab.plot(T,X,c="g")
# pylab.title("Time Signal Wave")
# pylab.xlabel("time (seconds)")
# pylab.ylabel("data")
# pylab.show()

'''使用xlwt，保存STFT数据'''
# angle_x = np.angle(x,deg=True)   #相位谱
angle_x=  np.arctan2(x.imag, x.real)*(180/np.pi)   #相位谱
print "angle_x:", angle_x.shape

x=abs(x)     #x若是复数，转换为实数
#使用workbook方法，创建一个新的工作簿book
book=xlwt.Workbook(encoding='utf-8')
#添加一个sheet
sheet=book.add_sheet('Sheet_1', cell_overwrite_ok=True)
for i,row in enumerate(x):
    for j,col in enumerate(row):
        sheet.write(i,j,col)
# print "'''开始保存数据为.xls'''"
# book.save('./语音图片1/data.xls')
# print "'''数据保存完成'''"

'''打印输出'''
print "f:\n",f.shape
print "t:\n",t.shape
print "x:\n",x.shape    #(513, 91)

'''画声谱图'''
plt.pcolormesh(t, f, np.abs(x))
# 查看此时的x轴的最大值和最小值和y轴的最大值最小值
# print "plt.axis:\n",plt.axis()
plt.ylim(0,4000)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
# plt.show()


