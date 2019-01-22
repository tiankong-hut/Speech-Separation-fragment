#coding=utf-8

import numpy as np
import wavio
import subprocess
from wave import Wave_read       #Wave_read 是一个类class

rate = 22050  # samples per second
T = 3         # sample duration (seconds)
f = 440.0     # sound frequency (Hz)
t = np.linspace(0, T, T*rate, endpoint=False)
x = np.sin(2*np.pi * f * t)
a=wavio.read("陌上花开.wav")
# print a.data.min()
# print a.data.max()
# wavio.read("sa1.wav")
# print a.data
# wavio.write("compose2.wav", a.data, 44100, sampwidth=2)  #生成wav文件
b=wavio.read("compose1.wav")
# print b
# subprocess.call([ '-i' 'sa1.mp3', 'sa1.wav'])
# print Wave_read.getsampwidth(a)

import matplotlib.pyplot as plt
# window=np.hamming(12)
# plt.plot(window)
# plt.title("Hamming window")
# plt.ylabel("Amplitude")
# plt.xlabel("Sample")
# plt.show()

from numpy.fft import fft, fftshift
# plt.figure()
# window = np.hamming(51)
# A = fft(window, 2048) / 25.5
# mag = np.abs(fftshift(A))
# freq = np.linspace(-0.5, 0.5, len(A))
# response = 20 * np.log10(mag)
# response = np.clip(response, -100, 100)
# plt.plot(freq, response)
# plt.title("Frequency response of Hamming window")
# plt.ylabel("Magnitude [dB]")
# plt.xlabel("Normalized frequency [cycles per sample]")
# plt.axis('tight')
# plt.show()

# 声谱图
fs = 1e4
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs           #arange（）表示1到100,000的数字
# print "time:\n",time.shape
freq = np.linspace(1e3, 2e3, N)
x = amp * np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
# print "x:\n",x.shape
# print "fs:\n",fs
# spectrogram:声谱图
from scipy import signal
f, t, Sxx = signal.spectrogram(x, fs)     #f-频率阵列，t-时间序列，SXX-x的声谱图
# print "f:\n",f.shape
# print "t:\n",t.shape
# print "Sxx:\n",Sxx[1:5,1:5]
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()


# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import colors as c
# X = np.linspace(0,1,100)
# Y = np.linspace(0,1,100)
# X,Y = np.meshgrid(X,Y)
# Z = (X**2 + Y**2) < 1.
# Z = Z.astype(int)
# Z += (X**2 + Y**2) < .5
# #不同的颜色
# cMap = c.ListedColormap(['y','b','m'])
# plt.pcolormesh(X,Y,Z,cmap=cMap)
# # plt.pcolormesh(X,Y,Z, cmap=plt.cm.Spectral)
# plt.show()


import wave
wf = wave.open("陌上花开.wav","rb")
params = wf.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
str_data = wf.readframes(nframes)
wf.close()
# 将波形数据转化为数组
s = np.fromstring(str_data, dtype=np.short)
# wave_data   这里波形是语音波形
s = np.fft.fft(s)  # fft  获得频谱
# print s.shape
# plt.plot(s,'g')
# plt.title('FFT', fontsize=9, color='r')
# plt.show()
# 这个要知道波形文件的采样率fs以及你的采样点数目N。一般而言，对普通的音频文件等，
# 频率间隔为fs/N，这是因为fs表示每秒钟采了多少点，除以总数N就恰好是频率间隔


# import numpy as np
# import pylab as pl
# import wave
# f = wave.open("/home/dyan/123.wav", "rb")
# # 读取格式信息
# # (nchannels, sampwidth, framerate, nframes, comptype, compname)
# params = f.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]
# # 读取波形数据
# str_data = f.readframes(nframes)
# f.close()
# wave_data = np.fromstring(str_data, dtype=np.short)
# # 在时间轴上画波形图
# # 以上nchannels=1, sampwidth=2, framerate=16000
# lenth=len(wave_data)
# ti=lenth/16000.0
# t = np.arange(0, ti, ti/lenth)
# pl.plot(t,wave_data)
# pl.show()

# https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile    # get the api
fs, data = wavfile.read('陌上花开.wav') # load the data
a = data.T[0]   # this is a two channel soundtrack, I get the first track
print a.shape
b=[(ele/2**8.)*2-1 for ele in a]   # this is 8-bit track, b is now normalized on [-1,1)
c = fft(b) # calculate fourier transform (complex numbers list)
d = len(c)/2    # you only need half of the fft list (real signal symmetry)
plt.plot(abs(c[:(d-1)]),'r')
plt.savefig('陌上花开'+'.png',bbox_inches='tight')
# plt.show()
## gcf: Get Current Figure
# fig = plt.gcf()
# plt.show()
# fig.savefig('test.png', dpi=100)