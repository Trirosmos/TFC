import pyaudio
from pynput import keyboard

from threading import Thread, Timer

from pitch_detecter_threads import crepe_get_audio, get_f0
from pitch_shifter_threads import pitch_shift, feedback, grava_queues
from voice_manager_threads import print_envelope_state, run_envelopes, midi_listen

from utils import delete_last_line

from consts import num_voices

p = pyaudio.PyAudio()

for i in range(0, 100):
	delete_last_line()

print_envelope_state()
run_envelopes()

t1 = Thread(target=crepe_get_audio, args = (p,))
t1.start()

t2 = Thread(target=get_f0)
t2.start()

for i in range(0, num_voices):
	t = Thread(target=pitch_shift, args = (p, i))
	t.start()

t3 = Thread(target=feedback, args = (p,))
t3.start()

t4 = Thread(target=midi_listen)
t4.start()

def on_press(key):
	global grava_audio
	global mutex
	try:
		if(key.char == "m"):    
			for q in grava_queues:
				q.put(True)
	except AttributeError:
		pass


# Collect events until released
with keyboard.Listener(
        on_press=on_press) as listener:
    listener.join()

t1.join()
t2.join()
t3.join()
t4.join()


p.terminate()



