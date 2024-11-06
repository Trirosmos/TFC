from threading import Timer
from utils import delete_last_line
from consts import num_voices

key_mapping = ["z", "s", "x", "d", "c", "v", "g", "b", "h", "n", "j", "m", #Primeira oitava
							 "q", "2", "w", "3", "e", "r", "5", "t", "6", "y", "7", "u", "i", "9", "o", "0", "p"] #Segunda oitava
key_state = []

envelope_minimum = -20

voice_state = []
voice_envelope = []
voice_note = []

attack = 0.2
decay = 0.2
sustain = 0.6
release = 0.5

envelope_update_period = 0.005 #Em segundos

for k in key_mapping:
	key_state.append(False)

for v in range(num_voices):
	voice_state.append("idle")
	voice_envelope.append(envelope_minimum)
	voice_note.append(0)

def print_envelope_state():
	global voice_envelope
	delete_last_line()
	print(voice_envelope)
	print(voice_note)
	Timer(0.05,print_envelope_state,[]).start()

def run_envelopes():
	global voice_state
	global voice_envelope
	global attack
	global decay
	global sustain
	global envelope_update_period
	global release

	for v in range(0, len(voice_state)):
		estado = voice_state[v]
		if(estado == "attack"):
			if(voice_envelope[v] >= 0):
				voice_envelope[v] = 0
				voice_state[v] = "decay"
			else:
				voice_envelope[v] += (-envelope_minimum) / (attack / envelope_update_period)
		
		if(estado == "decay"):
			if(voice_envelope[v] > sustain):
				voice_envelope[v] -= (1 - sustain) / (decay / envelope_update_period)
			else:
				voice_state[v] = "sustain"
		
		if(estado == "release"):
			if(voice_envelope[v] > envelope_minimum):
				voice_envelope[v] -= (-envelope_minimum) / (release / envelope_update_period)
			else:
				voice_envelope[v] = envelope_minimum
				voice_state[v] = "idle"

	Timer(envelope_update_period,run_envelopes,[]).start()

def inicia_nota(tecla):
	try:
		indice_nota = key_mapping.index(tecla)
		if(key_state[indice_nota] == False):
			try:
				idle_index = voice_state.index("idle")
				voice_state[idle_index] = "attack"
				voice_note[idle_index] = indice_nota
			except ValueError:
				pass
		key_state[indice_nota] = True
	except ValueError:
		pass

def finaliza_nota(tecla):
	try:
		indice_nota = key_mapping.index(tecla)
		if(key_state[indice_nota] == True):
			try:
				active_index = voice_note.index(indice_nota)
				voice_state[active_index] = "release"
			except ValueError:
				pass
		key_state[indice_nota] = False
	except ValueError:
		pass
	

def on_press(key):
	try:
		tecla = key.char
		inicia_nota(tecla)
	except AttributeError:
		pass

def on_release(key):
	try:
		tecla = key.char
		finaliza_nota(tecla)
	except AttributeError:
		pass