import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift
import sounddevice as sd
import time
import threading
from play import main

# print(sd.query_devices())

freqList = [697, 1209, 770, 1336, 852, 1477, 941, 1633]

def calcFFT(signal, fs):
    # https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    #y  = np.append(signal, np.zeros(len(signal)*fs))
    N  = len(signal)
    T  = 1/fs
    xf = np.linspace(-1.0/(2.0*T), 1.0/(2.0*T), N)
    yf = fft(signal)
    return(xf, fftshift(yf))

def generateSin(freq, time, fs):
    n = time*fs #numero de pontos
    x = np.linspace(0.0, time, n)  # eixo do tempo
    s = np.sin(freq*x*2*np.pi)
    return s

def analyzeFreqs(arr):
    solutions = []
    for i in arr:
        for ref in freqList:
            if abs(i-ref) <= 1 and not ref in solutions:
                solutions.append(ref)
    return sorted(solutions)

reverseSignalTable = {
    (697, 1209): 1, 
    (697, 1336): 2, 
    (697, 1477): 3, 
    (697, 1633): 'A', 
    (770, 1209): 4, 
    (770, 1336): 5, 
    (770, 1477): 6, 
    (770, 1633): 'B', 
    (852, 1209): 7, 
    (852, 1336): 8, 
    (852, 1477): 9, 
    (852, 1633): 'C', 
    (941, 1209): 'X', 
    (941, 1336): 0, 
    (941, 1477): '#', 
    (941, 1633): 'D'
    }

T = 10
fs = 44100
t = np.linspace(0, T, T*fs)

rec = sd.rec(int(T*fs), fs, channels = 2, device = 8)
sd.wait()
rec = rec[:,0]


# print(rec)
# print("\n")


plt.plot(np.linspace(0, T, T*fs), rec, '.-')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Recording")
plt.show()

fftx, ffty = calcFFT(rec, fs)

plt.plot(fftx, np.abs(ffty), '.-')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"FFT of Rec")
plt.show()

import peakutils
index = peakutils.indexes(np.abs(ffty), thres=0.8, min_dist=50)
print("index de picos {}" .format(index))
for freq in fftx[index]:
    print("freq de pico sao {}Hz" .format(freq))

foundFreqs = abs(fftx[index])
try:
    a,b = analyzeFreqs(foundFreqs)
    print(f'\n=================\nThe detected number for that signal is {reverseSignalTable[(a,b)]}\n=================\n')
except:
    print("Sorry, no number could be detected from that signal.")