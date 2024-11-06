import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided
from scipy.io import wavfile

import pyaudio
from pylibrb import RubberBandStretcher, Option, create_audio_array
import librosa

from threading import Thread
import queue

import time

from consts import sr, samples_per_block, amostras_bloco, amostras_pitch, LEN

from pitch_detect_threads import crepe_get_audio, get_f0, crepe_pitch_out_queue
from pitch_shifter_threads import pitch_shift, feedback

p = pyaudio.PyAudio()

print(samples_per_block)

t1 = Thread(target=crepe_get_audio, args = (p,))
t1.start()

t2 = Thread(target=get_f0)
t2.start()

t3 = Thread(target=pitch_shift, args = (p,))
t3.start()

t4 = Thread(target=feedback, args = (p,))
t4.start()

t1.join()
t2.join()
t3.join()
t4.join()
p.terminate()

