import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import wavfile
import librosa

from crepe import predict, predict_tpu

try:
	from crepe import interpreter
except:
	pass

single = False

if(single):
	nomes_musicas = ["01-AchGottundHerr"]
	nomes_instrumentos = ["-violin.wav", "-clarinet.wav", "-saxphone.wav", "-bassoon.wav"]
else:
	nomes_musicas = ["01-AchGottundHerr", "02-AchLiebenChristen", "03-ChristederdubistTagundLicht", "04-ChristeDuBeistand",
"05-DieNacht", "06-DieSonne", "07-HerrGott", "08-FuerDeinenThron", "09-Jesus", "10-NunBitten"]

	#nomes_musicas = ["01-AchGottundHerr", "02-AchLiebenChristen"]
	nomes_instrumentos = ["-violin.wav", "-clarinet.wav", "-saxphone.wav", "-bassoon.wav"]

f0s = []
estimativas_yin = []
estimativas_crepe = []
estimativas_crepe_quantizado = []

erros_crepe = np.empty(1)
erros_crepe_quantizado = np.empty(1)
erros_yin = np.empty(1)

midi_0 = 440 * ((2**(1/12)) ** (-69))

for m in nomes_musicas:
	f0 = loadmat("../Bach10_v1.1/" + m + "/" + m + "-GTF0s.mat")["GTF0s"]
	f0 = 440 * ((2**(1/12)) ** (-69 + f0))
	for t in f0:
		f0s.append(t)

for m in nomes_musicas:
	for i in nomes_instrumentos:
		print("Processando arquivo " + str(len(estimativas_yin) + 1))
		sr, y = wavfile.read("../Bach10_v1.1/" + m + "/" + m + i)
		y = y.astype("float64")

		_, estimativa_crepe, _, _ = predict(y, sr, "medium", viterbi = True, center = False, step_size=10, verbose=0)
		estimativa_crepe = np.array(estimativa_crepe).squeeze()

		try:
			_, estimativa_crepe_quantizado, _, _ = predict_tpu(y, sr, interpreter, viterbi=True, center = False, step_size=10, verbose=0)
			estimativa_crepe_quantizado = np.array(estimativa_crepe_quantizado).squeeze()
		except:
			pass

		estimativa_yin = librosa.yin(y, sr = sr, frame_length = int(0.064 / (1/sr)), trough_threshold = 0.1, fmin = 65, fmax = 2093, hop_length = int(0.01 / (1/sr)), center = False)
		min_len = np.minimum(len(estimativa_crepe), len(estimativa_yin))

		estimativa_yin = estimativa_yin[:min_len]
		estimativa_crepe = estimativa_crepe[:min_len]

		try:
			estimativa_crepe_quantizado = estimativa_crepe_quantizado[:min_len]
		except:
			pass

		estimativas_crepe.append(estimativa_crepe)
		estimativas_yin.append(estimativa_yin)

		try:
			estimativas_crepe_quantizado.append(estimativa_crepe_quantizado)
		except:
			pass

def calcular_erro(true, pred):
	true = np.array(true)
	pred = np.array(pred)

	min_len = np.minimum(len(true), len(pred))
	true = true[:min_len]
	pred = pred[:min_len]

	erro = pred / true
	erro = 100 * np.log(erro)/np.log(2**(1/12))
	erro[np.where(true < 10)] = 0
	erro = np.sort(erro)
	return erro

for m in range(len(f0s)):
	try:
		erros_crepe_quantizado = np.concatenate((erros_crepe_quantizado, calcular_erro(f0s[m], estimativas_crepe_quantizado[m])))
	except:
		pass

	erros_crepe = np.concatenate((erros_crepe, calcular_erro(f0s[m], estimativas_crepe[m])))
	erros_yin = np.concatenate((erros_yin, calcular_erro(f0s[m], estimativas_yin[m])))

np.savez("erros.npz", yin = np.array(erros_yin), crepe = np.array(erros_crepe), crepe_quant = np.array(erros_crepe_quantizado))

tempo = len(f0s[0])/100
tempos = np.linspace(0, tempo, len(f0s[0]))
tempos = tempos[:len(estimativas_yin[0])]

fig, axs = plt.subplots(3, 1, sharex = True, sharey = True)

axs[0].set_title("Comparação: estimadores de frequência fundamental")

axs[2].set_xlabel("Tempo (s)")


axs[0].set_ylabel("Frequência (Hz)")
axs[1].set_ylabel("Frequência (Hz)")
axs[2].set_ylabel("Frequência (Hz)")

axs[0].plot(tempos, f0s[0][:len(estimativas_yin[0])], color = "green", label = "Anotação")
axs[0].plot(tempos, estimativas_yin[0], color = "red", alpha = 0.3, label = "YIN")
axs[0].legend()

try:
	axs[2].plot(tempos, f0s[0][:len(estimativas_yin[0])], color = "green", label = "Anotação")
	axs[2].plot(tempos, estimativas_crepe_quantizado[0], color = "blue", alpha = 0.3, label = "CREPE Quantizado")
	axs[2].legend()
except:
	pass

axs[1].plot(tempos, f0s[0][:len(estimativas_yin[0])], color = "green", label = "Anotação")
axs[1].plot(tempos, estimativas_crepe[0], color = "orange", alpha = 0.6, label = "CREPE")
axs[1].legend()


plt.show()