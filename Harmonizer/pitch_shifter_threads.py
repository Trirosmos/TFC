import queue
import pyaudio

from pitch_detecter_threads import crepe_pitch_out_queues
from consts import sr, amostras_bloco, LEN
from pylibrb import RubberBandStretcher, Option, create_audio_array

import numpy as np
import time

from voice_manager_threads import voice_envelope, voice_note

def feedback(p):
	stream = p.open(
		format=pyaudio.paFloat32, channels=1, rate=sr, input=True, frames_per_buffer = amostras_bloco
	)
	player = p.open(
    format=pyaudio.paFloat32, channels=1, rate=sr, output=True, frames_per_buffer = amostras_bloco
	)

	while(True):
		novo_bloco = np.frombuffer(stream.read(amostras_bloco), dtype=np.float32).reshape(-1, ) * 0.6
		player.write(novo_bloco, amostras_bloco)

	stream.stop_stream()
	stream.close()

def pitch_shift(p, v):
	global voice_envelope
	global voice_note

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

	while(True):
		while(stretcher.available() < amostras_bloco):
			data = np.expand_dims(np.frombuffer(stream.read(amostras_bloco), dtype=np.float32), axis = 0) * np.exp(voice_envelope[v])
			stretcher.process(data, False)  

			try:
				f0 = crepe_pitch_out_queues[v].get(False)
			except queue.Empty:
				pass

			if(voice_note[v] != -1):
				freq = 440 * ((2**(1/12))**(-69 + 12 + voice_note[v]))
				if(f0[0] > 0 and f0[1] > 0.3):
					stretcher.pitch_scale = freq/f0[0]

		saida = stretcher.retrieve(amostras_bloco)                
		player.write(saida, amostras_bloco)

	stream.stop_stream()
	stream.close()