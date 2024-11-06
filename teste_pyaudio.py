import pyaudio
import numpy as np
from pylibrb import RubberBandStretcher, Option, create_audio_array

import time
from threading import Thread

import os

def delete_last_line():
    "Use this function to delete the last line in the STDOUT"

    #cursor up one line
    os.sys.stdout.write('\x1b[1A')

    #delete last line
    os.sys.stdout.write('\x1b[2K')

CHUNK = 128
RATE = 48000
LEN = 60

p = pyaudio.PyAudio()

def run_audio(ratio):
	stream = p.open(
    format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK
	)
	player = p.open(
    format=pyaudio.paFloat32, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK
	)
	stretcher = RubberBandStretcher(sample_rate=RATE,
                                channels=1,
                                options=Option.PROCESS_REALTIME | Option.ENGINE_FINER | Option.WINDOW_STANDARD | Option.FORMANT_PRESERVED ,
                                initial_pitch_scale=ratio)
	stretcher.set_max_process_size(CHUNK)
	pad = create_audio_array(channels_num=1, samples_num=stretcher.get_preferred_start_pad())
	stretcher.process(pad)
	delay = stretcher.get_start_delay()

	for i in range(int(LEN * RATE / CHUNK)):  # go for a LEN seconds
		inicio = time.time()
		while(stretcher.available() < CHUNK):
			fim = time.time()
			data = np.expand_dims(np.frombuffer(stream.read(CHUNK), dtype=np.float32), axis = 0)
			stretcher.process(data, False)  

		saida = stretcher.retrieve(CHUNK)                
		player.write(saida, CHUNK)

	stream.stop_stream()
	stream.close()
	
for i in range(0, 100):
    delete_last_line()

t1 = Thread(target=run_audio, args=(0.25,))
t2 = Thread(target=run_audio, args=(0.5,))
t3 = Thread(target=run_audio, args=(2,))
t4 = Thread(target=run_audio, args=(1,))

t1.start()
t2.start()
t3.start()
t4.start()

t1.join()
t2.join()
t3.join()
t4.join()
p.terminate()
