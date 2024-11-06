import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided
from scipy.io import wavfile

import pyaudio

from pycoral.utils import edgetpu

import librosa

from crepe_tpu import predict_tpu, predict

from threading import Thread
import queue

import time

sr = 48000
LEN = 60

amostras_pitch = 1024

samples_per_block = int(amostras_pitch * (sr / 16000))

q = queue.Queue()

def feedback():
	stream = p.open(
		format=pyaudio.paFloat32, channels=1, rate=sr, input=True, frames_per_buffer = 128
	)
	player = p.open(
    format=pyaudio.paFloat32, channels=1, rate=sr, output=True, frames_per_buffer = 128
	)

	for i in range(int(LEN * sr / 128)):
		novo_bloco = np.frombuffer(stream.read(128), dtype=np.float32).reshape(-1, )
		player.write(novo_bloco, 128)

	stream.stop_stream()
	stream.close()

def get_audio():
	stream = p.open(
		format=pyaudio.paFloat32, channels=1, rate=sr, input=True, frames_per_buffer = samples_per_block
	)

	for i in range(int(LEN * sr / samples_per_block)):
		novo_bloco = np.frombuffer(stream.read(samples_per_block), dtype=np.float32).reshape(-1, )
		q.put(novo_bloco)

	stream.stop_stream()
	stream.close()


def get_f0():
	interpreter = edgetpu.make_interpreter("crepe_medium_edgetpu.tflite")
	interpreter.allocate_tensors()

	confidence_threshold = 0.3

	while(True):
		novo_bloco = q.get(block = True, timeout = 30)

		inicio = time.time()
		#_, frequency, confidence, activation = predict(taken_chunk, sr, "medium", viterbi=True, center=True, step_size=(latencia_pitch * 1000), verbose=0)
		_, frequency, confidence, activation = predict_tpu(novo_bloco, sr, interpreter, viterbi=True, center=False, step_size = int(1000/(16000 / amostras_pitch)), verbose=0)
		fim = time.time()
		if(confidence[0] >= confidence_threshold):
			print(frequency, confidence, "Tempo: " + str((fim - inicio) * 1000 ))

p = pyaudio.PyAudio()

t1 = Thread(target=get_audio)
t1.start()

t2 = Thread(target=get_f0)
t2.start()

t3 = Thread(target=feedback)
t3.start()

t1.join()
t2.join()
t3.join()
p.terminate()

