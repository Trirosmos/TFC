import numpy as np
import matplotlib.pyplot as plt

attack = 0.2
decay = 0.5
sustain = 0.1
release = 0.4

envelope_minimum = -20

envelope_update_period = 0.005 #Em segundos

voice_state = []
voice_envelope = []
voice_note = []

num_voices = 1

for v in range(num_voices):
	voice_state.append("idle")
	voice_envelope.append(envelope_minimum)
	voice_note.append(-1)

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

voice_state[0] = "attack"

envelope_out = []
tempo_envelope = []
momentos_transicao = []

sustain_counter = 0

for i in range(int(1 / 0.005)):
	envelope_out.append(np.exp(voice_envelope[0]))
	last_state = voice_state[0]
	run_envelopes()

	if(voice_state[0] == "sustain"):
		sustain_counter += 1
		if(sustain_counter == 20):
			voice_state[0] = "release"

	new_state = voice_state[0]

	if(last_state != new_state):
		momentos_transicao.append(i + 1)

	tempo_envelope.append(i * 0.005)
	
	if(voice_envelope[0] == envelope_minimum):
		break

print(np.log(sustain) / (decay / envelope_update_period))

print(momentos_transicao)

plt.plot(tempo_envelope[:momentos_transicao[0]], envelope_out[:momentos_transicao[0]], color = "red", label = "Attack")
plt.plot(tempo_envelope[momentos_transicao[0] - 1:momentos_transicao[1]], envelope_out[momentos_transicao[0] - 1:momentos_transicao[1]], color = "green", label = "Decay")
plt.plot(tempo_envelope[momentos_transicao[1] - 1:momentos_transicao[2]], envelope_out[momentos_transicao[1] - 1:momentos_transicao[2]], color = "blue", label = "Sustain")
plt.plot(tempo_envelope[momentos_transicao[2] - 1:], envelope_out[momentos_transicao[2] - 1:], color = "orange", label = "Release")
plt.legend()
plt.title("Sinal do gerador de envoltória")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude normalizada")
plt.show()