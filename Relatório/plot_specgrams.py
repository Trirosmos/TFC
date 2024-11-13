import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile

hop_length = 128

sr1, y1 = wavfile.read("e4_164.8138Hz_voz_base.wav")
y1 = np.array(y1).astype("float64").transpose()[0]

sr2, y2 = wavfile.read("e4_164.8138Hz_voz_1.wav")
y2 = np.array(y2).astype("float64").transpose()[0]

D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1, hop_length=hop_length)),ref=np.max)
D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2, hop_length=hop_length)),ref=np.max)

diff = np.abs(D2 - D1)

def plot_dois():
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

    img1 = librosa.display.specshow(D1, y_axis='log', sr=sr1, hop_length=hop_length, x_axis='time', ax=ax[0])
    img2 = librosa.display.specshow(D2, y_axis='log', sr=sr1, hop_length=hop_length, x_axis='time', ax=ax[1])

    ax[0].set_xlim(0,3)

    ax[0].set_ylabel("Frequência (Hz)")
    ax[1].set_ylabel("Frequência (Hz)")
    ax[0].set_xlabel("Tempo (s)")
    ax[1].set_xlabel("Tempo (s)")

    ax[0].set(title='Espectrograma do sinal natural')
    ax[1].set(title='Espectrograma do sinal sintetizado')

    ax[0].label_outer()
    ax[1].label_outer()


    fig.colorbar(img1, ax=ax, format="%+2.f dB")
    fig.colorbar(img2, ax=ax, format="%+2.f dB")
    plt.show()

def plot_um():
    plt.close()
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    img1 = librosa.display.specshow(diff, y_axis='log', sr=sr1, hop_length=hop_length, x_axis='time', ax=ax)
    ax.set_ylabel("Frequência (Hz)")
    ax.set_xlabel("Tempo (s)")
    ax.set(title='Diferença entre espectrogramas')
    ax.label_outer()
    fig.colorbar(img1, ax=ax, format="%+2.f dB")
    ax.set_xlim(0,3)
    plt.show()
    
plot_dois()
