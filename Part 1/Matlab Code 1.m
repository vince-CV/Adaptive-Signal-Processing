clc;
clear all;
close all;

% generate and display input random white gaussian noise
L=10000;
mu=0;

sigma_v=sqrt(1);
v=sigma_v*randn(L,1)+mu;
sigma_s=sqrt(10);
s=sigma_s*randn(L,1)+mu;

% FIR filters
g=[1,1];
h=[1,2,1];

x=conv(s,g);
x=x(1:10000);
t=conv(x,h);
d=t(1:10000)+v;

X = fft(x);
X = abs(X);
X= X.*X/length(X);
D = fft(d);
D = abs(D);
D = D.*D/length(D);

nperseg=256;
noverlap=nperseg/2;
nfft=nperseg;
[Pxx,W] = cpsd(x,x,hann(nperseg),noverlap,nfft,'twosided'); % cpsd() normalizes window rms value
[Pdx,W] = cpsd(d,x,hann(nperseg),noverlap,nfft,'twosided'); % cpsd() normalizes window rms value

freq=(0:1:nfft-1)/nfft;
n=0:1:L-1;
figure()
subplot(2,1,1)
plot(x);title('input signal x[n]');ylabel('x');xlabel('n');grid
subplot(2,1,2)
plot(d);title('input signal d[n]');ylabel('d');xlabel('n');grid

figure()
subplot(2,1,1)
plot(X);title('FFT |X[k]|^2/N');ylabel('X');xlabel('k');grid; % ax=axis; axis([ax(1) ax(2) ax(3) 10.0]) % example adjusting the axis scale
subplot(2,1,2)
plot(D);title('FFT |D[k]|^2/N');ylabel('D');xlabel('k');grid; % ax=axis; axis([ax(1) ax(2) ax(3) 10.0]) % example adjusting the axis scale

figure()
subplot(2,1,1)
plot(freq, Pxx);title('PSD Pxx[f]');ylabel('Pxx');xlabel('f');grid; % ax=axis;axis([ax(1) ax(2) 0.0 ax(4)]) % example adjusting the axis scale
subplot(2,1,2)
plot(freq, Pdx);title('PSD Pdx[f]');ylabel('Pdx');xlabel('f');grid; % ax=axis;axis([ax(1) ax(2) 0.0 ax(4)]) % example adjusting the axis scale

W=Pdx/Pxx;
w=ifft(W);


%% w(1:3)=0.8456 1.2764 0.8475
