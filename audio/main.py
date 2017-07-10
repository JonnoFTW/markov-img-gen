import sounddevice as sd
from scipy.io import wavfile
from scipy import signal
import sys
import matplotlib.pyplot as plt
import os
import subprocess as sp
from collections import defaultdict, Counter
import pyprind
import numpy as np
import random
from .utils import readwav

output_seconds = 5
n_gram = 2
fname = 'Night_And_Day.flac'
fs, data, outname = readwav(fname)


def show_pcm(d):
    x = np.arange(d.shape[0])
    plt.plot(x, d)
    plt.ylabel('PCM')
    plt.xlabel('Time [sec]')
    plt.show()


def show_spectrogram(d):
    # from matplotlib.pyplot import specgram
    # specgram(d, NFFT=256, Fs=fs)
    f, t, Sxx = signal.spectrogram(d, fs, nperseg=256)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def show_fft(d):
    fft = np.fft.fft(d)
    x = np.arange(d.shape[0])
    plt.plot(x, fft)
    plt.ylabel('FFT')
    plt.xlabel('Time [sec]')
    plt.show()


sample = data[fs * 70:fs * 80]
wavfile.write('sample.wav', fs, sample)
# sd.play(sample, fs, blocking=True)
# show_pcm(data)
# show_fft(data)
show_spectrogram(data)
if input("cont: ").lower() == 'n':
    exit()

output_frames = fs * output_seconds
print(fs)


# data = np.fft.fft(data)

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


# for each step, take each pair an

transitionsL = defaultdict(Counter)
# transitionsR = defaultdict(Counter)

bs = 1
# sd.play(sample, fs, blocking=True)
print("Training")
prog = pyprind.ProgBar(sample.shape[0], stream=1)
for gram in find_ngrams(sample, n_gram):
    # learn the difference between each step given a history

    transitionsL[gram[0]][gram[1] - gram[0]] += 1
    prog.update()
    # transitionsR[gram[0][1]][gram[1][1]] += 1

generated = np.zeros_like(data[:output_frames])
# choose a random starting frame from the transition table
# stateL = np.random.choice(list(transitionsL.keys()))
all_states = list(transitionsL.keys())
stateL = random.choice(all_states)

prog = pyprind.ProgBar(output_frames + 1, width=64, stream=1)

print("\nGenerating")
restarts = 0
for i in range(output_frames):
    node = transitionsL[stateL]
    if len(node) == 0:
        restarts += 1
        stateL = random.choice(all_states)
        node = transitionsL[stateL]
    counts = np.array(list(node.values()), dtype=np.float32)
    keys = list(node.keys())
    key_idxs = np.arange(len(keys))
    ps = counts / counts.sum()
    col_idx = np.random.choice(key_idxs, p=ps)
    generated[i] = stateL + keys[col_idx]
    generated[i] *= bs
    stateL = stateL + keys[col_idx]
    # stateR
    prog.update()

print("Restarts={}".format(restarts / output_frames))
# generated = np.fft.ifft(generated).real
print("\nPlaying")
all_frames = np.concatenate((sample, generated))
sd.play(generated, fs, blocking=True)
print("Finished playing")
wavfile.write(outname, fs, all_frames)
print(sp.check_output([ffmpeg, '-i', outname, '-vn',
                       '-ar', '44100', '-ac', '2', '-ab', '192k', '-y', '-f', 'mp3', outname[:-4] + '.mp3']))
