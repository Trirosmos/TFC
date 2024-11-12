from threading import Timer
from utils import delete_last_line
from consts import num_voices
import rtmidi
import numpy as np

active_notes = []
envelope_minimum = -20

voice_state = []
voice_envelope = []
voice_note = []

attack = 0.02
decay = 0.2
sustain = 0.6
release = 0.02

envelope_update_period = 0.005 #Em segundos

midiin = rtmidi.RtMidiIn()

for v in range(num_voices):
	voice_state.append("idle")
	voice_envelope.append(envelope_minimum)
	voice_note.append(-1)

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
				voice_envelope[v] += np.log(sustain) / (decay / envelope_update_period)
			else:
				voice_envelope[v] += (-envelope_minimum) / (attack / envelope_update_period)
		
		if(estado == "decay"):
			if(np.exp(voice_envelope[v]) > sustain):
				voice_envelope[v] += np.log(sustain) / (decay / envelope_update_period)
			else:
				voice_state[v] = "sustain"
		
		if(estado == "release"):
			if(voice_envelope[v] > envelope_minimum):
				voice_envelope[v] -= (-envelope_minimum) / (release / envelope_update_period)
			else:
				voice_envelope[v] = envelope_minimum
				voice_state[v] = "idle"

	Timer(envelope_update_period,run_envelopes,[]).start()


def inicia_nota(indice_nota):
	try:
		idle_index = voice_state.index("idle")
		voice_state[idle_index] = "attack"
		voice_note[idle_index] = indice_nota

		delete_last_line()
		delete_last_line()
		print("Recebeu nota " + str(indice_nota))
	except ValueError:
		pass

def finaliza_nota(indice_nota):
	try:
		active_index = voice_note.index(indice_nota)

		voice_note[active_index] = -1
		voice_state[active_index] = "release"

		delete_last_line()
		delete_last_line()
		print("Recebeu nota " + str(indice_nota))
	except ValueError:
		pass
	
def midi_listen():
	ports = range(midiin.getPortCount())
	if ports:
		for i in ports:
			print(midiin.getPortName(i))
		print("Opening port 0!") 
		midiin.openPort(0)
		midiin.openPort(1)
		while True:
			m = midiin.getMessage(250) # some timeout in ms
			if m:
				if(m.isNoteOn()):
					inicia_nota(m.getNoteNumber())
				if(m.isNoteOff()):
					finaliza_nota(m.getNoteNumber())
	else:
		print("Nenhuma entrada MIDI encontrada")