# coding=utf-8
# http://blog.csdn.net/ouening/article/details/71079535
# http://blog.sina.com.cn/s/blog_40793e970102w3m2.html
# http://blog.csdn.net/qq_28006327/article/details/59129110?winzoom=1
# # 有很多工具方便地支持STFT展开，可以使用scipy库中的signal模块。
# # 做STFT分解,音频信号（wav文件）的路径存在path变量中
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html#scipy.signal.spectrogram

from __future__ import division     #必须放在最前面，division进行精确的除法运算
import matplotlib.pyplot as plt
import wavio
import numpy as np
from scipy import signal
import wave
import pylab
import xlwt

# f = wave.open('sa1.wav', 'rb')   #双通道
# f = wave.open('陌上花开.wav', 'rb')   #双通道
# params = f.getparams()     #最多6种参数，parameter-参数
# nchannels, sampwidth, framerate, nframes = params[:4]
# print nchannels, sampwidth, framerate, nframes

# wav_s=wavio.read("陌上花开.wav")
wav_s=wavio.read("TIMIT-sa1-white.wav")
# wav_s=wavio.read("sa1.wav")
# wav_s=wavio.read("white.wav")   #白噪声
print "wav_struct:\n", wav_s
# Wav(data.shape=(793800, 2), data.dtype=int16, rate=44100, sampwidth=2)
# N/rate=793800/44100=18s

# time 是一个列表，与wave_data[0]或wave_data[1]配对形成系列点坐标
# time = data/rate= 793800/44100= 18s
time = np.arange(0,wav_s.data.shape[0])*(1.0/wav_s.rate)

# '''' 绘制时间的波形图'''''
# pylab.subplot(311)
# pylab.plot(time, wav_s.data[:,0],c="g")
# pylab.title("Time Signal Wave")
# pylab.xlabel("time (seconds)")
# pylab.ylabel("data[:,0]")

# pylab.subplot(312)
# pylab.plot(time, wav_s.data[:,1], c="g")
# pylab.xlabel("time (seconds)")
# pylab.ylabel("data[:,1]")

pylab.show()


# 采样点数，修改采样点数和起始位置进行不同位置和长度的音频波形分析
N=wav_s.data.shape[0]     # N=793800     # data/rate= 793800/44100= 18s
start=0 #开始采样位置
print "rate:", wav_s.rate
df = wav_s.rate/(N-1) # 分辨率: 0.05556 ,用division：精确到小数点后面几位
print "分辨率:",df

freq = []       #列表初始化
freq = [df*k for k in range(0, N)]   #列表解析
print "len(freq):",len(freq)         #793800
# print "freq:", freq                  #范围：0-44100Hz

# 或者：
# freq = []       #列表初始化
# for n in range(0,N):
#     freq_1 = n*df
#     freq.append(freq_1)   #附加到列表末尾


wav_data=wav_s.data[:,0][start:start+N]
complex_S=np.fft.fft(wav_data)     #IFFT的时候用,同时用于神经网络输入矩阵：
input_value = complex_S            #用于神经网络输入矩阵
print " input_value: ", input_value.shape[0]

'''''
将FFT的数据写入到excel文件中
'''''
# input_value = input_value[0:len(input_value)]
# convert list to array
data_array = np.array(input_value)
print "data_array:", type(data_array)
print "data_array:", data_array[0:20]

'''保存数据为txt格式'''
# np.savetxt('./语音图片/data_1.txt', data_array, delimiter=',')
# np.savetxt('./语音图片/data_1.txt', data_array, fmt='%.5f')    #取小数点后5位

'''保存数据为xls格式'''
#使用workbook方法，创建一个新的工作簿book
book=xlwt.Workbook(encoding='utf-8')
#添加一个sheet
sheet=book.add_sheet('Sheet_1', cell_overwrite_ok=True)
for i,row in enumerate(data_array):
    for j,col in enumerate(row):
        sheet.write(i,j,col)
book.save('./语音图片/data.xls')
print 'Finish saving xls file'

complex=np.fft.fft(wav_data)*2/N
# print wav_s.data    #整数
print "complex: ",complex.shape    #
'''常规显示采样频率一半的频谱'''
d=int(len(complex)/2)
# print d           #396900
'''仅显示频率在4000以下的频谱'''
# while freq[d]>3000:
#   d-=100
# pylab.plot(freq[:d-1],abs(complex[:d-1]),'r')  #abs-取模值
# pylab.xlabel("Frequency(Hz)")
# pylab.ylabel("Amplitude")
# pylab.show()

''''IFFT: 傅里叶反变换'''
Data=np.fft.ifft(complex_S)
print "Data.shape:",Data.shape
# pylab.subplot(313)
# pylab.plot(time, Data)
# pylab.title("IFFT: Time Signal Wave")
# pylab.xlabel("time (seconds)")
# pylab.ylabel("data[:,0]")
# pylab.show()



# print "wav_struct.data:\n",wav_struct.data[:,0]      #’：‘表示所有行,0和1分别代表第一列和第二列
wav=wav_s.data[:,0].astype(float)/np.power(2,wav_s.sampwidth*8-1)   #astype-变量类型转换，power-幂指数
print "wav:\n",wav.shape
# Wav(data.shape=(793800, 2), data.dtype=int16, rate=44100, sampwidth=2)
[f,t,X]=signal.spectral.spectrogram(wav, 44100, np.hamming(1024), nperseg=1024,
               noverlap=0, detrend=False, return_onesided=True, mode='magnitude')
# [f,t,X]=signal.spectral.spectrogram(wav,44100)       #频率要限制在3000HZ内

print "f:\n",f.shape
print "t:\n",t.shape
print "X:\n",X.shape

# spectrogram:声谱图
plt.pcolormesh(t, f, X)
# 查看此时的x轴的最大值和最小值和y轴的最大值最小值
print "plt.axis:\n",plt.axis()
# 设置四个最值
# plt.axis([-100, 100, 0, 100])  #（不要，必须去掉）

plt.ylim(0,2500)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()



# f, t, Sxx = signal.spectrogram(x, fs)
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# '''
# 使用梅尔滤波器组得到梅尔频率谱
# '''
# import numpy as np
# import librosa
#
# melW=librosa.filters.mel(fs=44100,n_fft=1024,n_mels=40,fmin=0.,fmax=22100)
# melW /= np.max(melW,axis=-1)[:,None]
# melX = np.dot(X,melW.T)
#
# # 记住一句话，在梅尔频谱上做倒谱分析（取对数，做DCT变换）就得到了梅尔倒谱
# import librosa
# melM = librosa.feature.mfcc(wav,sr=44100,n_mfcc=20)


# http://nbviewer.jupyter.org/url/jakevdp.github.io/downloads/notebooks/UnderstandingTheFFT.ipynb
# import numpy as np
# import timeit
# def DFT_slow(x):
#     """Compute the discrete Fourier Transform of the 1D array x"""
#     x = np.asarray(x, dtype=float)
#     N = x.shape[0]
#     n = np.arange(N)
#     k = n.reshape((N, 1))
#     M = np.exp(-2j * np.pi * k * n / N)
#     return np.dot(M, x)
# 通过与Numpy内置的FFT函数进行比较，我们可以对结果进行二次检查：
# 在此，我们将快速检查我们的算法是否产生了正确的结果：
# allclose():如果两个数组元素在公差范围内相等，则返回true
# x = np.random.random(1024)
# print np.allclose(DFT_slow(x), np.fft.fft(x))
# True
# print np.allclose([1e10,1e-7], [1.00001e10,1e-8])
# False
# 只是为了确认我们的算法的惯性，我们比较了这两种方法的执行时间：

'''__name__ 是当前模块名，当模块被直接运行时模块名为 __main__'''
# 这句话的意思就是，当模块被直接运行时，以下代码块将被运行，当模块是被导入时，代码块不被运行。
# if __name__=='__main__':
#     print timeit.timeit(stmt="DFT_slow(0.1)", setup="from __main__ import DFT_slow")

   # from timeit import Timer
   # t1=Timer(("DFT_slow(0.3)", "from __main__ import DFT_slow")
   # print t1.timeit(1000)

# 官方文档：
# To give the timeit module access to functions you define,
# you can pass a setup parameter which contains an import statement:
# def test():
#     """Stupid test function"""
#     L = []
#     for i in range(100):
#         L.append(i)
# if __name__ == '__main__':
#     import timeit
#     print(timeit.timeit("test()", setup="from __main__ import test"))


# print timeit.timeit("DFT_slow(0.01)", "from __main__ import DFT_slow")      #timeit-python中计时工具
# print timeit.timeit("np.fft.fft(x)")
# 10 loops, best of 3: 75.4 ms per loop
# 10000 loops, best of 3: 25.5 µs per loop

# print timeit.timeit('"-".join([str(n) for n in range(100)])', number=10000)
# 1.31257581711





# DFT到FFT: 利用对称性
# def FFT(x):
#     """A recursive implementation of the 1D Cooley-Tukey FFT"""
#     x = np.asarray(x, dtype=float)
#     N = x.shape[0]
#
#     if N % 2 > 0:
#         raise ValueError("size of x must be a power of 2")
#     elif N <= 32:  # this cutoff should be optimized
#         return DFT_slow(x)
#     else:
#         X_even = FFT(x[::2])
#         X_odd = FFT(x[1::2])
#         factor = np.exp(-2j * np.pi * np.arange(N) / N)
#         return np.concatenate([X_even + factor[:N / 2] * X_odd,
#                                X_even + factor[N / 2:] * X_odd])
#
# # 在此，我们将快速检查我们的算法是否产生了正确的结果：
# x = np.random.random(1024)
# np.allclose(FFT(x), np.fft.fft(x))
# True
# 算法执行时间长度
# %timeit DFT_slow(x)
# %timeit FFT(x)
# %timeit np.fft.fft(x)
# 10 loops, best of 3: 77.6 ms per loop
# 100 loops, best of 3: 4.07 ms per loop
# 10000 loops, best of 3: 24.7 µs per loop