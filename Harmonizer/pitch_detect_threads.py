from crepe_tpu import predict_tpu, predict
import time
from pycoral.utils import edgetpu

import queue
import pyaudio

import numpy as np

from consts import sr, samples_per_block, amostras_bloco, amostras_pitch, LEN

crepe_pitch_out_queue = queue.Queue()
crepe_audio_queue = queue.Queue()

def crepe_get_audio(p):
	stream = p.open(
		format=pyaudio.paFloat32, channels=1, rate=sr, input=True, frames_per_buffer = samples_per_block
	)

	for i in range(int(LEN * sr / samples_per_block)):
		novo_bloco = np.frombuffer(stream.read(samples_per_block), dtype=np.float32).reshape(-1, )
		crepe_audio_queue.put(novo_bloco)

	stream.stop_stream()
	stream.close()

def get_f0():
	interpreter = edgetpu.make_interpreter("crepe_medium_edgetpu.tflite")
	interpreter.allocate_tensors()

	while(True):
		novo_bloco = crepe_audio_queue.get(block = True, timeout = 30)
		equivalente_1024_amostras = int(1024 * (sr/16000))

		if(len(novo_bloco) < equivalente_1024_amostras):
			novo_bloco = np.concatenate((novo_bloco, np.zeros(equivalente_1024_amostras - len(novo_bloco))))

		inicio = time.time()
		#_, frequency, confidence, activation = predict(taken_chunk, sr, "medium", viterbi=True, center=True, step_size=(latencia_pitch * 1000), verbose=0)
		_, frequency, confidence, activation = predict_tpu(novo_bloco, sr, interpreter, viterbi=True, center=False, step_size = int(1000/(16000 / amostras_pitch)), verbose=0)
		fim = time.time()
		crepe_pitch_out_queue.put([frequency[0], confidence[0]])