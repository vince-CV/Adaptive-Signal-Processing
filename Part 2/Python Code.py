import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy import fftpack
from scipy import signal
from scipy import random
from scipy import io
from scipy import linalg

def stft(x):
    M=np.empty([1,1024])
    for i in range (0, 500):
        s=128*i
        m=x[s:s+1024]*np.hanning(1024)
        mf=scipy.fft(m)
        if i==0:
            M=mf
        else:
            M=np.vstack((M,mf))
    return M.T

ratex, x = io.wavfile.read('x.wav')
rated, d = io.wavfile.read('d.wav')

Zxx = stft(x)
Zdd = stft(d)

corr_xx_n =[]
corr_xd_n = []
corr_dd_n = []
wopt_n1 = []
MMSE_n1 = []
R_n1 = []
p_n1 = []
d_n1 = []
wopt_n1 = []
norm_MMSE_n1 = []
norm_MMSE_n1_dB = []
dB_n1 = np.empty(1024)

###Q1
for i in range(0,1024):
    corr_xx = np.correlate(Zxx[i,:], Zxx[i,:], mode='full')/len(Zxx[i,:])
    corr_xx = corr_xx[len(Zxx[i,:])-1:]
    
    corr_xd = np.correlate(Zxx[i,:], Zdd[i,:], mode='full')/len(Zxx[i,:])
    corr_xd = corr_xd[len(Zxx[i,:])-1:]
    
    corr_dd = np.correlate(Zdd[i,:], Zdd[i,:], mode='full')/len(Zxx[i,:])
    corr_dd = corr_dd[len(Zxx[i,:])-1:]
    
    R_n1 = scipy.linalg.toeplitz(corr_xx[0], corr_xx[0])
    p_n1 = corr_xd[0]
    d_n1 = corr_dd[0]
    
    wopt_n1 = 1/R_n1 * p_n1
    MMSE_n1 = d_n1 - p_n1.conj() * wopt_n1
    norm_MMSE_n1 = np.real(MMSE_n1 / d_n1)
    #dB_n1[i] = norm_MMSE_n1
    norm_MMSE_n1_dB = 10*np.log10(norm_MMSE_n1)
    dB_n1[i] = norm_MMSE_n1_dB

plt.plot(dB_n1[0:512])
plt.grid()
plt.show()

### N = 5
dB_n5 = np.empty(1024)
d_n5_n = np.empty(1024)
print ('when N = 5')
for j in range(0,513):
    corr_xx = np.correlate(Zxx[j,:], Zxx[j,:], mode='full')/len(Zxx[j,:])
    corr_xx = corr_xx[len(Zxx[j,:])-1:]
    
    corr_xd = np.correlate(Zxx[j,:], Zdd[j,:], mode='full')/len(Zxx[j,:])

    corr_dd = np.correlate(Zdd[j,:], Zdd[j,:], mode='full')/len(Zdd[j,:])
    corr_dd = corr_dd[len(Zxx[j,:])-1:]
    
    R_n5 = scipy.linalg.toeplitz(corr_xx[0:5].conj(), corr_xx[0:5])
    p_n5 = corr_xd[(len(Zxx[j,:])-1):(len(Zxx[j,:])-6):-1]

    d_n5 = corr_dd[0]
    wopt_n5 = np.matmul(np.linalg.inv(R_n5) , p_n5)
    MMSE_n5 = np.real(d_n5 - np.matmul(p_n5.conj().T, wopt_n5))
    norm_MMSE_n5 = MMSE_n5 / np.real(d_n5)
    #dB_n5[j] = norm_MMSE_n5
    norm_MMSE_n5_dB = 10*np.log10(norm_MMSE_n5)
    dB_n5[j] = norm_MMSE_n5_dB

plt.plot(dB_n5[0:512])
plt.grid()
plt.show()

#Q2

L=len(Zxx)
sum_n = np.empty(L)
norm_MMSE_n1_dB_n=np.empty(L)

for i in range(0,1024):
    corr_xx = (np.correlate(Zxx[i,:], Zxx[i,:], mode='full')/len(Zxx[j,:]))[len(Zxx[i,:])-1:]
    
    corr_xd = (np.correlate(Zxx[i,:], Zdd[i,:], mode='full')/len(Zxx[j,:]))[len(Zdd[i,:])-1:]
    
    corr_dd = (np.correlate(Zdd[i,:], Zdd[i,:], mode='full')/len(Zxx[j,:]))[len(Zdd[i,:])-1:]
    
    R_n1 = scipy.linalg.toeplitz(corr_xx[0], corr_xx[0])
    p_n1 = corr_xd[0]
    wopt_n1 = 1/R_n1 * p_n1
    d_n1 = np.matrix(Zdd[i,:])
    
    e_n1 = d_n1 - np.matmul(wopt_n1.conj(), np.matrix(Zxx[i,:]))
 
    sum_e=0
    for j in range(0,500):
        sum_e = sum_e + e_n1[0,j]*e_n1[0,j].conj()
        
    MMSE_n1 = sum_e/500

    norm_MMSE_n1 = MMSE_n1 / corr_dd[0]

    norm_MMSE_n1_dB_n[i] = 10*np.log10(norm_MMSE_n1)

plt.plot(norm_MMSE_n1_dB_n[0:512])
plt.show()  





dB_n5 = np.empty(L)
norm_MMSE_n5_dB_n=np.empty(L)
norm_MMSE_n51=np.empty(L)
for j in range(0, 1024):
    
    corr_xx = (np.correlate(Zxx[j,:], Zxx[j,:], mode='full')/len(Zxx[j,:]))[len(Zxx[j,:])-1:]
    
    corr_xd = (np.correlate(Zxx[j,:], Zdd[j,:], mode='full')/len(Zxx[j,:]))

    corr_dd = (np.correlate(Zdd[j,:], Zdd[j,:], mode='full')/len(Zxx[j,:]))[len(Zxx[j,:])-1:]
  
    R_n5 = scipy.linalg.toeplitz(corr_xx[0:5].conj(), corr_xx[0:5])
    p_n5 = corr_xd[(len(Zxx[j,:])-1):(len(Zxx[j,:])-6):-1]
    d_n5 = np.matrix(Zdd[j,:])
    
    wopt_n5 = np.linalg.inv(R_n5).dot((p_n5).T)

    M=np.zeros(shape=(5,500))
    m1=Zxx[j,:]
    m2=np.hstack((np.zeros(1),Zxx[j,:]))[0:500]
    m3=np.hstack((np.zeros(1),m2))[0:500]
    m4=np.hstack((np.zeros(1),m3))[0:500]
    m5=np.hstack((np.zeros(1),m4))[0:500]
    m=np.vstack((np.vstack((np.vstack((np.vstack((m1,m2)),m3)),m4)),m5))
    
    e_n5 = d_n5 - np.matmul(wopt_n5.conj(), m)
 
    sum_e=0

    for mm in range(0,500):
        sum_e = sum_e + e_n5[0,mm]*e_n5[0,mm].conj()
        
    MMSE_n5 = sum_e/500

    norm_MMSE_n5 = MMSE_n5 / corr_dd[0]
    
    norm_MMSE_n51[j]=norm_MMSE_n5
    norm_MMSE_n5_dB = 10*np.log10(np.real(norm_MMSE_n5))
    norm_MMSE_n5_dB_n[j] = norm_MMSE_n5_dB

plt.plot(norm_MMSE_n5_dB_n[0:512])
plt.show()  


# Q3

seg_yn=np.empty(shape=(1024,500),dtype='complex')

for i in range(0,1024):
    corr_xx = (np.correlate(Zxx[i,:], Zxx[i,:], mode='full')/len(Zxx[j,:]))[len(Zxx[i,:])-1:]
    
    corr_xd = (np.correlate(Zxx[i,:], Zdd[i,:], mode='full')/len(Zxx[j,:]))[len(Zdd[i,:])-1:]
    
    corr_dd = (np.correlate(Zdd[i,:], Zdd[i,:], mode='full')/len(Zxx[j,:]))[len(Zdd[i,:])-1:]
    
    R_n1 = scipy.linalg.toeplitz(corr_xx[0], corr_xx[0])
    p_n1 = corr_xd[0]
    wopt_n1 = 1/R_n1 * p_n1
    
    y_n = np.matmul(wopt_n1.conj(), np.matrix(Zxx[i,:]))
    seg_yn[i:]=y_n

    
for j in range(0,500):
    
    y_n=np.real(fftpack.ifft(seg_yn[:,j]))
    seg_yn[:,j]=y_n

size=len(x)
yn=np.zeros(size)

for segm in range(0,500):
    for post in range(0,1024):
        yn[segm*128+post]=seg_yn[post,segm]+yn[segm*128+post]

en=d-yn/4

plt.plot(en)
plt.show()
plt.grid()

length=len(en)
e=en[896:length-896]
dd=d[896:length-896]
leng=len(e)
plt.plot(e)
plt.show()

sum_d=np.zeros(leng)
MMSE=np.zeros(leng)
for idx in range(0,leng-1):
    MMSE[idx]=e[idx]**2
    sum_d[idx]=dd[idx]**2
norm_MMSE=np.sum(MMSE)/np.sum(sum_d)
norm_MMSE_dB=10*np.log10(np.real(norm_MMSE))
print(norm_MMSE)

seg_yn=np.empty(shape=(1024,500),dtype='complex')


for j in range(0, 1024):
    
    corr_xx = (np.correlate(Zxx[j,:], Zxx[j,:], mode='full')/len(Zxx[j,:]))[len(Zxx[j,:])-1:]
    
    corr_xd = (np.correlate(Zxx[j,:], Zdd[j,:], mode='full')/len(Zxx[j,:]))

    corr_dd = (np.correlate(Zdd[j,:], Zdd[j,:], mode='full')/len(Zxx[j,:]))[len(Zxx[j,:])-1:]
  
    R_n5 = scipy.linalg.toeplitz(corr_xx[0:5].conj(), corr_xx[0:5])
    p_n5 = corr_xd[(len(Zxx[j,:])-1):(len(Zxx[j,:])-6):-1]
    d_n5 = np.matrix(Zdd[j,:])
    
    wopt_n5 = np.linalg.inv(R_n5).dot((p_n5).T)

    M=np.zeros(shape=(5,500))
    m1=Zxx[j,:]
    m2=np.hstack((np.zeros(1),Zxx[j,:]))[0:500]
    m3=np.hstack((np.zeros(1),m2))[0:500]
    m4=np.hstack((np.zeros(1),m3))[0:500]
    m5=np.hstack((np.zeros(1),m4))[0:500]
    m=np.vstack((np.vstack((np.vstack((np.vstack((m1,m2)),m3)),m4)),m5))
    
    y_n = np.matmul(wopt_n5.conj(), m)
    seg_yn[j:]=y_n
    
for k in range(0,500):
    y_n=np.real(fftpack.ifft(seg_yn[:,k]))
    seg_yn[:,k]=y_n

size=len(x)
yn=np.zeros(size)

for segm in range(0,500):
    for post in range(0,1024):
        yn[segm*128+post]=seg_yn[post,segm]+yn[segm*128+post]

en=d-yn/4

plt.plot(en)
plt.show()
plt.grid()

length=len(en)
e=en[896:length-896]
dd=d[896:length-896]
leng=len(e)
plt.plot(e)
plt.show()

MMSE=np.zeros(leng)
sum_d=np.zeros(leng)
for idx in range(0,leng-1):
    MMSE[idx]=e[idx]**2
    sum_d[idx]=dd[idx]**2
norm_MMSE=np.sum(MMSE)/np.sum(sum_d)
norm_MMSE_dB=10*np.log10(np.real(norm_MMSE))
print(norm_MMSE)





