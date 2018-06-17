#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:47:50 2018

@author: Xunzhe Wen
"""
import scipy
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal
from scipy import random
from scipy import linalg
import scipy.io.wavfile
from scipy import io
import scipy.io.wavfile
from scipy import linalg
from scipy import io

ratex, x = io.wavfile.read('x.wav')
rated, d = io.wavfile.read('d.wav')

x=np.float64(x)
d=np.float64(d)

corr_xx = np.zeros(3)
corr_xx[0] = np.correlate(x, x)/len(x)
corr_xx[1] = np.correlate(np.hstack((x,[0])), np.hstack(([0],x)))/len(x)
corr_xx[2] = np.correlate(np.hstack((x,[0,0])), np.hstack(([0,0],x)))/len(x)

corr_xd = np.zeros(3)
corr_xd[0] = np.correlate(x, d)/len(d)
corr_xd[1] = np.correlate(np.hstack(([0],x)), np.hstack((d,[0])))/len(d)
corr_xd[2] = np.correlate(np.hstack(([0,0],x)), np.hstack((d,[0,0])))/len(d)

power_d = np.correlate(d,d)/len(d)
power_x = np.correlate(x,x)/len(x)

R = np.matrix(scipy.linalg.toeplitz(corr_xx[0:3].conj(), corr_xx[0:3]))
p = np.matrix(corr_xd).T

wopt = np.matmul(np.linalg.inv(R), p)

mu_1 =np.float(0.25/(3 * power_x))
mu_2 =0.25

w_1=np.array([0,0,0]).reshape(3,1)
w_2=np.array([0,0,0]).reshape(3,1)

MSE_1=np.empty(500)
MSE_2=np.empty(500)
a=[]

for i in range(0,500):
    MSE_SD=(power_d-2*(w_1.T).dot(p)+(w_1.T).dot(R).dot(w_1))/power_d
    MSE_NT=(power_d-2*(w_2.T).dot(p)+(w_2.T).dot(R).dot(w_2))/power_d

    MSE_1[i]=MSE_SD
    MSE_2[i]=MSE_NT
    
    w_1 = w_1 - mu_1*(R.dot(w_1)-p)
    a.append(w_1)
    w_2 = w_2 - mu_2*linalg.inv(R).dot(R.dot(w_2)-p)


plt.title('MMSE over iterations')
plt.xlabel('Iterations')
plt.ylabel('MMSE')
plt.plot(MSE_1,label='Steepest Descent')
plt.plot(MSE_2,label='Newton')
plt.grid()
plt.legend()
plt.show()

l=len(x)
y=np.convolve(w_2.A1,x)
e=d-y[:l]

io.wavfile.write('e.wav', ratex, np.int16(e))
