import math
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy import pi
from scipy.fftpack import fft


def u(t1):
    return 1 * (t1 >= 0)

B=1024
N=(3)*B
t = np.linspace(0,3 ,12*B)
frequency= np.linspace(0, 512,np.int(N/2))

fn1=np.random.randint(0,512)
fn2=np.random.randint(0,512)
waveform1 = np.sin(2*pi*fn1*t)
waveform2 = np.sin(2*pi*fn2*t)
noise= waveform1+waveform2


F=190.83
f=200.65
C=130.8
c=261.63
D=146.83
d=293.66
A=55.0
a=110.0
Freq1=[F,F,F,C,D,D,C,A,F,F,F,C,D,D,C]
Freq2=[f,f,f,c,d,d,c,a,f,f,f,c,d,d,c]
ti=[0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.61,1.8,2,2.21,2.4,2.6,2.8]
Ti=0.2
S=0
for i in range(len(Freq1)):
    X=np.sin(2*np.pi*Freq1[i]*t)+np.sin(2*np.pi*Freq2[i]*t)
    u1=u(t-ti[i])
    u2=u(t-ti[i]-Ti)
    S=S + np.multiply(X,u1-u2)
plt.subplot(3,2,1)
plt.plot(t,S)
#plt.show()

plt.title('Time Domain Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
#plt.show()
peaks=2/N * np.abs( fft(S) [0:np.int(N/2)])
plt.subplot(3,2,2)
#plt.figure()
plt.plot(frequency, peaks)
plt.title('freq Domain Signal')
plt.xlabel('freq')
plt.ylabel('Amplitude')
#plt.show()
S=S+noise 

freq_data=fft(S) 

y=2/N * np.abs(freq_data[0:np.int(N/2)])  
maxx=np.ceil(max(peaks))
count=2
for j in range(len(frequency)):
   if(count==0):
      break
   if( (y[j]>maxx) and (count==2)):
       count=count-1
       f1=frequency[j]
      
   elif( (y[j]>maxx) and (count==1)):
        count=count-1
        f2=frequency[j]    
    
noiseF=np.sin(2*pi*np.int(f1)*t)+np.sin(2*pi*np.int(f2)*t)


plt.subplot(3,2,4)
#plt.figure()
plt.plot(frequency, y)
plt.title('freq Domain Signal with noise')
plt.xlabel('freq')
plt.ylabel('Amplitude')
plt.subplot(3,2,3)
plt.plot(t,S)
#plt.show()
x_filtered=S-noiseF
plt.subplot(3,2,5)
#plt.figure()
plt.plot(t, x_filtered)
sd.play(x_filtered, 3*1024)
freq_filtered=fft(x_filtered) 

y_f=2/N * np.abs(freq_filtered[0:np.int(N/2)])  
plt.subplot(3,2,6)
#plt.figure()
plt.plot(frequency, y_f)
plt.title('freq Domain Signal after noise cancellation')
plt.xlabel('freq')
plt.ylabel('Amplitude')
plt.show()