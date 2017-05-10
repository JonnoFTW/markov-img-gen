from scipy.io import wavfile
import os
import subprocess as sp
import numpy as np

def readwav(fname):
    """
    Reads in 2 channel file and converts it to mono PCM
    :param fname: 
    :return: the file as an array of pcm data 
    """
    wavname = fname.split('.')[0] + '.wav'
    outname = 'generated.' + wavname

    ffmpeg = 'ffmpeg'
    if not os.path.exists(wavname):
        print(sp.check_output([ffmpeg, '-i', fname, wavname]))

    fs, data = wavfile.read(wavname)
    data = (data.sum(axis=1) / 2).astype(np.int16)
    return fs, data, outname


def train_test_split(x, y, test_size=0.33):
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must both have same number of rows")
    split_idx = int(x.shape[0] * (1 - test_size))
    return x[:split_idx], x[split_idx:], y[:split_idx], y[split_idx:]




