import queue
import pyaudio

from pitch_detect_threads import crepe_pitch_out_queue
from consts import sr, samples_per_block, amostras_bloco, amostras_pitch, LEN
from pylibrb import RubberBandStretcher, Option, create_audio_array

import numpy as np
import time

def feedback(p):
	stream = p.open(
		format=pyaudio.paFloat32, channels=1, rate=sr, input=True, frames_per_buffer = amostras_bloco
	)
	player = p.open(
    format=pyaudio.paFloat32, channels=1, rate=sr, output=True, frames_per_buffer = amostras_bloco
	)

	for i in range(int(LEN * sr / amostras_bloco)):
		novo_bloco = np.frombuffer(stream.read(amostras_bloco), dtype=np.float32).reshape(-1, )
		player.write(novo_bloco, amostras_bloco)

	stream.stop_stream()
	stream.close()

def pitch_shift(p):
	stream = p.open(
		format=pyaudio.paFloat32, channels=1, rate=sr, input=True, frames_per_buffer = amostras_bloco
	)
	player = p.open(
    format=pyaudio.paFloat32, channels=1, rate=sr, output=True, frames_per_buffer = amostras_bloco
	)

	stretcher = RubberBandStretcher(sample_rate=sr,
                                channels=1,
                                options=Option.PROCESS_REALTIME | Option.ENGINE_FINER | Option.WINDOW_STANDARD | Option.FORMANT_PRESERVED ,
                                initial_pitch_scale=1)
	stretcher.set_max_process_size(amostras_bloco)
	pad = create_audio_array(channels_num=1, samples_num=stretcher.get_preferred_start_pad())
	stretcher.process(pad)
	delay = stretcher.get_start_delay()

	f0 = [110, 1]
	last_timestamp = time.time()

	for i in range(int(LEN * sr / amostras_bloco)):
		while(stretcher.available() < amostras_bloco):
			data = np.expand_dims(np.frombuffer(stream.read(amostras_bloco), dtype=np.float32), axis = 0)
			stretcher.process(data, False)  
			if(f0[0] > 0 and f0[1] > 0.3):
				stretcher.pitch_scale = 110/f0[0]

			try:
				f0 = crepe_pitch_out_queue.get(False)
			except queue.Empty:
				pass

		saida = stretcher.retrieve(amostras_bloco)                
		player.write(saida, amostras_bloco)

	stream.stop_stream()
	stream.close()