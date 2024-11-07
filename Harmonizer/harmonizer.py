import pyaudio
from pynput import keyboard

from threading import Thread, Timer

from pitch_detecter_threads import crepe_get_audio, get_f0
from pitch_shifter_threads import pitch_shift, feedback
from voice_manager_threads import print_envelope_state, run_envelopes, on_press, on_release

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

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

t1.join()
t2.join()
t3.join()
p.terminate()



