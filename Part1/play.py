def main(): 
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal
    from scipy.fftpack import fft, fftshift
    import sounddevice as sd
    import time

    # print(sd.query_devices())

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

        
    def convertToFreq(pair, T = 5, fs = 44100):
        f1, f2 = pair
        return (generateSin(f1, T, fs) + generateSin(f2, T, fs))

    signalTable = {
        1:  (697,1209),
        2:  (697,1336),
        3:  (697,1477),
        "A":(697,1633),
        4:  (770,1209),
        5:  (770,1336),
        6:  (770,1477),
        "B":(770,1633),    
        7:  (852,1209),
        8:  (852,1336),
        9:  (852,1477),
        "C":(852,1633),
        "X":(941,1209),
        0:  (941,1336),
        "#":(941,1477),
        "D":(941,1633)
    }


    print("Starting! \n")
    chosen = False
    while not chosen:
        inp = input("\nEscolha um número para enviar (0 a 9): ")
        chosen = inp in signalTable.keys()
        if not chosen:
            chosen = int(inp) in signalTable.keys()
        if not chosen:
            print("Opção inválida, tente novamente!\n")
    # inp = 7
    selection = int(inp)
    print(f"\n================================\nSeleção concluída com sucesso!\nVocê escolheu o número: {selection}")

    T = 1
    fs = 44100
    t = np.linspace(0, T, T*fs)
    y = convertToFreq(signalTable[selection], T, fs)

    sd.play(y, fs, device=3)
    sd.wait()

    print("\nFazendo gráfico")
    plt.plot(t, y, '.-')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Graph for input number {selection} composed of {signalTable[selection]}Hz")
    plt.xlim((0,1/300))
    plt.show()
    print("\nGráfico concluído!\n Tocando som! \n")


if __name__ == "__main__":
    main()
