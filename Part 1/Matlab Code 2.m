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

% Unbiased estimate
[UE_xx lag_xx]=xcorr(x,x,'unbiased');
[UE_dx lag_dx]=xcorr(d,x,'unbiased');

figure()
plot(lag_xx, UE_xx);title('Unbiased Auto-Correlation of x[n]');ylabel('Estimation');xlabel('Lags');grid
figure()
plot(lag_dx, UE_dx);title('Unbiased Cross-Correlation of x[n] and d[n]');ylabel('Estimation');xlabel('Lags');grid

pos=find(lag_xx==0);
xx=UE_xx(pos:pos+2);

pos=find(lag_dx==0);
dx=UE_dx(pos:pos+2);


