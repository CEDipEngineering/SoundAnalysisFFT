import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift
import sounddevice as sd
import soundfile as sf

def calcFFT(signal, fs):
    # https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    #y  = np.append(signal, np.zeros(len(signal)*fs))
    N  = len(signal)
    T  = 1/fs
    xf = np.linspace(-1.0/(2.0*T), 1.0/(2.0*T), N)
    yf = fft(signal)
    return(xf, np.abs(fftshift(yf)))

def generateSin(freq, time, fs):
    n = time*fs #numero de pontos
    x = np.linspace(0.0, time, n)  # eixo do tempo
    s = np.sin(freq*x*2*np.pi)
    return s

def normalizeArray(arr):
    # https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
    # Normalize between [-1,1]
    return 2.*(arr - np.min(arr))/np.ptp(arr)-1

def LPF(signal, cutoff_hz, fs):
        from scipy import signal as sg
        #####################
        # Filtro
        #####################
        # https://scipy.github.io/old-wiki/pages/Cookbook/FIRFilter.html
        nyq_rate = fs/2
        width = 5.0/nyq_rate
        ripple_db = 120.0 #dB
        N , beta = sg.kaiserord(ripple_db, width)
        taps = sg.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
        return(sg.lfilter(taps, 1.0, signal))

def modulateSignal(signal, fs):
    T = len(signal)//fs
    carrier = generateSin(14000, T, fs)

    return signal * carrier

playRec = False
playNormalized = False
playFiltered = False
playModulated = False
playDemodulated = True

data, fs = sf.read("Part2/data/recording.wav")
T = len(data)//fs
t = np.linspace(0, T,T*fs)

data = data[:,0]
dataNormalized = normalizeArray(data)
dataFiltered = LPF(dataNormalized, 4000, fs)
dataModulated = modulateSignal(dataFiltered, fs)
dataDemodulated = modulateSignal(dataModulated, fs)

print("Started ffts")
dataFFT = calcFFT(data, fs)
dataNormalizedFFT = calcFFT(dataNormalized, fs)
dataFilteredFFT = calcFFT(dataFiltered, fs)
dataModulatedFFT = calcFFT(dataModulated, fs)
dataDemodulatedFFT = calcFFT(dataDemodulated, fs)
print("Transforms applied")

if playRec:
    sd.play(data, fs, device=3)
    sd.wait()

if playNormalized:
    sd.play(dataNormalized, fs, device=3)
    sd.wait()

if playFiltered:
    sd.play(dataFiltered, fs, device=3)
    sd.wait()

if playModulated:
    sd.play(dataModulated, fs, device=3)
    sd.wait()

if playDemodulated:
    sd.play(dataDemodulated, fs, device=3)
    sd.wait()


# Plotting time domain

plt.subplot(5,1,1)
plt.title('Original message')
plt.plot(t, data,'g')
plt.ylabel('Amplitude')
plt.xlabel('time (s)')

plt.subplot(5,1,2)
plt.title('Message normalized between [-1,1]')
plt.plot(t, dataNormalized)
plt.ylabel('Amplitude')
plt.xlabel('time (s)')

plt.subplot(5,1,3)
plt.title('Message filtered to remove frequencies above 4kHz')
plt.plot(t, dataFiltered)
plt.ylabel('Amplitude')
plt.xlabel('time (s)')

plt.subplot(5,1,4)
plt.title('Message modulating carrier of 14kHz')
plt.plot(t, dataModulated)
plt.ylabel('Amplitude')
plt.xlabel('time (s)')

plt.subplot(5,1,5)
plt.title('Demodulated message')
plt.plot(t, dataDemodulated)
plt.ylabel('Amplitude')
plt.xlabel('time (s)')

plt.subplots_adjust(hspace=1)
plt.rc('font', size=15)
fig = plt.gcf()
fig.set_size_inches(16, 9)

fig.savefig('Part2/AM Time.png', dpi=160)

# Now for the graphs of FFT

print("Starting plot 1")
plt.subplot(5,1,1, label="test")
plt.title('Original message')
plt.plot(dataFFT[0], dataFFT[1],'g')
plt.ylabel('Amplitude')
plt.xlabel('frequency (Hz)')

print("Starting plot 2")
plt.subplot(5,1,2, label="test2")
plt.title('Message normalized between [-1,1]')
plt.plot(dataNormalizedFFT[0], dataNormalizedFFT[1])
plt.ylabel('Amplitude')
plt.xlabel('frequency (Hz)')

print("Starting plot 3")
plt.subplot(5,1,3, label="test3")
plt.title('Message filtered to remove frequencies above 4kHz')
plt.plot(dataFilteredFFT[0], dataFilteredFFT[1])
plt.ylabel('Amplitude')
plt.xlabel('frequency (Hz)')

print("Starting plot 4")
plt.subplot(5,1,4, label="test4")
plt.title('Message modulating carrier of 14kHz')
plt.plot(dataModulatedFFT[0], dataModulatedFFT[1])
plt.ylabel('Amplitude')
plt.xlabel('frequency (Hz)')

plt.subplot(5,1,5, label="test5")
plt.title('Demodulated message')
plt.plot(dataDemodulatedFFT[0], dataDemodulatedFFT[1])
plt.ylabel('Amplitude')
plt.xlabel('frequency (Hz)')

plt.subplots_adjust(hspace=1)
plt.rc('font', size=15)
fig = plt.gcf()
fig.set_size_inches(16, 9)

fig.savefig('Part2/AM Fourier.png', dpi=160)