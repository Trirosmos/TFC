import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided
from scipy.io import wavfile

import time

from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
import pycoral

import librosa

# store as a global variable, since we only support a few models for now
models = {
    'tiny': None,
    'small': None,
    'medium': None,
    'large': None,
    'full': None
}

# the model is trained on 16kHz audio
model_srate = 16000

def build_and_load_model(model_capacity):
    """
    Build the CNN model and load the weights

    Parameters
    ----------
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity, which determines the model's
        capacity multiplier to 4 (tiny), 8 (small), 16 (medium), 24 (large),
        or 32 (full). 'full' uses the model size specified in the paper,
        and the others use a reduced number of filters in each convolutional
        layer, resulting in a smaller model that is faster to evaluate at the
        cost of slightly reduced pitch estimation accuracy.

    Returns
    -------
    model : tensorflow.keras.models.Model
        The pre-trained keras model loaded in memory
    """
    from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization
    from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense
    from tensorflow.keras.models import Model

    if models[model_capacity] is None:
        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]

        layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        x = Input(shape=(1024,), name='input', dtype='float32')
        y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

        for l, f, w, s in zip(layers, filters, widths, strides):
            y = Conv2D(f, (w, 1), strides=s, padding='same',
                       activation='relu', name="conv%d" % l)(y)
            y = BatchNormalization(name="conv%d-BN" % l)(y)
            y = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid',
                          name="conv%d-maxpool" % l)(y)
            y = Dropout(0.25, name="conv%d-dropout" % l)(y)

        y = Permute((2, 1, 3), name="transpose")(y)
        y = Flatten(name="flatten")(y)
        y = Dense(360, activation='sigmoid', name="classifier")(y)

        model = Model(inputs=x, outputs=y)

        package_dir = "CREPE/"
        filename = "model-{}.h5".format(model_capacity)
        model.load_weights(os.path.join(package_dir, filename))
        model.compile('adam', 'binary_crossentropy')

        models[model_capacity] = model

    return models[model_capacity]

def to_local_average_cents(salience, center=None):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.cents_mapping = (
                np.linspace(0, 7180, 360) + 1997.3794084376191)

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / (weight_sum + 2.22044604925e-16)
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.
    """
    from hmmlearn import hmm

    # uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the
    # others
    self_emission = 0.1
    emission = (np.eye(360) * self_emission + np.ones(shape=(360, 360)) *
                ((1 - self_emission) / 360))

    # fix the model parameters because we are not optimizing the model
    model = hmm.CategoricalHMM(360, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = \
        starting, transition, emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                     range(len(observations))])


def get_activation(audio, sr, model_capacity='full', center=True, step_size=10,
                   verbose=1):
    """

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : int
        Set the keras verbosity mode: 1 (default) will print out a progress bar
        during prediction, 0 will suppress all non-error printouts.

    Returns
    -------
    activation : np.ndarray [shape=(T, 360)]
        The raw activation matrix
    """
    model = build_and_load_model(model_capacity)

    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    audio = audio.astype(np.float32)
    if sr != model_srate:
        # resample audio if necessary
        from resampy import resample
        audio = resample(audio, sr, model_srate)

    # pad so that frames are centered around their timestamps (i.e. first frame
    # is zero centered).
    if center:
        audio = np.pad(audio, 512, mode='constant', constant_values=0)

    # make 1024-sample frames of the audio with hop length of 10 milliseconds
    hop_length = int(model_srate * step_size / 1000)
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()

    # normalize each frame -- this is expected by the model
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)

    # run prediction and convert the frequency bin weights to Hz
    output = model.predict(frames, verbose=verbose)
    return output

def get_activation_tpu(audio, sr, instancia, center=True, step_size=10,
                   verbose=1):
            
	if len(audio.shape) == 2:
		audio = audio.mean(1)  # make mono
	audio = audio.astype(np.float32)
	if sr != model_srate:
		# resample audio if necessary
		from resampy import resample
		audio = resample(audio, sr, model_srate)
          
	# pad so that frames are centered around their timestamps (i.e. first frame
	# is zero centered).
	if center:
		audio = np.pad(audio, 512, mode='constant', constant_values=0)

	# make 1024-sample frames of the audio with hop length of 10 milliseconds
	hop_length = int(model_srate * step_size / 1000)
	n_frames = 1 + int((len(audio) - 1024) / hop_length)
	frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
	frames = frames.transpose().copy()

	# normalize each frame -- this is expected by the model
	frames -= np.mean(frames, axis=1)[:, np.newaxis]
	frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)
  
	output = []
	output_details = instancia.get_output_details()[0]
	input_details = instancia.get_input_details()[0]
     
	instancia.resize_tensor_input(input_details['index'], [len(frames), 1024])
	instancia.allocate_tensors()
     
	instancia.set_tensor(input_details['index'], frames)
	instancia.invoke()
	output = instancia.get_tensor(output_details['index'])
	return output


def predict(audio, sr, model_capacity='full',
            viterbi=False, center=True, step_size=10, verbose=1):
    """
    Perform pitch estimation on given audio

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : int
        Set the keras verbosity mode: 1 (default) will print out a progress bar
        during prediction, 0 will suppress all non-error printouts.

    Returns
    -------
    A 4-tuple consisting of:

        time: np.ndarray [shape=(T,)]
            The timestamps on which the pitch was estimated
        frequency: np.ndarray [shape=(T,)]
            The predicted pitch values in Hz
        confidence: np.ndarray [shape=(T,)]
            The confidence of voice activity, between 0 and 1
        activation: np.ndarray [shape=(T, 360)]
            The raw activation matrix
    """
    activation = get_activation(audio, sr, model_capacity=model_capacity,
                                center=center, step_size=step_size,
                                verbose=verbose)

    confidence = activation.max(axis=1)

    if viterbi:
        cents = to_viterbi_cents(activation)
    else:
        cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0

    return time, frequency, confidence, activation

def predict_tpu(audio, sr, instancia,
            viterbi=False, center=True, step_size=10, verbose=1):
    
    activation = get_activation_tpu(audio, sr, instancia,
                                center=center, step_size=step_size,
                                verbose=verbose)
    confidence = activation.max(axis=1)

    if viterbi:
        cents = to_viterbi_cents(activation)
    else:
        cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0

    return time, frequency, confidence, activation

directory = os.fsencode("Representative Dataset/")
    
def get_pitch(audio, sr):
	tempo, freq, conf, act = predict(audio, sr, "medium", viterbi=True, center=True, step_size=10, verbose=1)
	plt.plot(tempo, freq, color="blue", label = "Saída CPU", alpha = 0.3)
    
def get_pitch_tpu(audio, sr, instancia):
	tempo, freq, conf, act = predict_tpu(audio, sr, instancia, viterbi=True, center=True, step_size=10, verbose=1)
	plt.plot(tempo, freq, color = "red", label = "Saída TPU", alpha = 0.3)

interpreter = edgetpu.make_interpreter("crepe_medium_edgetpu.tflite")
interpreter.allocate_tensors()

#escolhido = random.choice(os.listdir(directory))
#escolhido = os.fsdecode(escolhido)
#print(escolhido)

def delete_last_line():
    "Use this function to delete the last line in the STDOUT"

    #cursor up one line
    os.sys.stdout.write('\x1b[1A')

    #delete last line
    os.sys.stdout.write('\x1b[2K')
    
for i in range(100):
	delete_last_line()
     
def compara_input_aleatorio():
	media_tpu = []
	media_cpu = []
	tamanhos_entrada = []
     
	for c in range(0, 20):
		medidas_tpu = []
		medidas_cpu = []
		tamanho_vetor_entrada = int(1024 + ((6144 - 1024)/20) * c)
          
		for t in range(0, 5):
			entrada = (np.random.rand(tamanho_vetor_entrada) * 2) - 1
               
			predict(entrada, 16000, "medium", viterbi=True, center=True, step_size=10, verbose=0)
			inicio = time.perf_counter()
			predict(entrada, 16000, "medium", viterbi=True, center=True, step_size=10, verbose=0)
			fim = time.perf_counter()
			print("Tempo CPU: " + str(fim - inicio))
               
			medidas_cpu.append(fim - inicio)

			predict_tpu(entrada, 16000, interpreter, viterbi=True, center = False, step_size=10, verbose=0)
			inicio = time.perf_counter()
			predict_tpu(entrada, 16000, interpreter, viterbi=True, center = False, step_size=10, verbose=0)
			fim = time.perf_counter()
			print("Tempo TPU: " + str(fim - inicio))
			print("Duração da entrada: " + str(tamanho_vetor_entrada / 16000))
               
			medidas_tpu.append(fim - inicio)
                  
		media_cpu.append(np.mean(np.array(medidas_cpu)))           
		media_tpu.append(np.mean(np.array(medidas_tpu)))
		tamanhos_entrada.append(c * 1024)
		print("Tempos calculados para entrada de tamanho " + str(tamanho_vetor_entrada))
		print("\n")
          
	return media_cpu, media_tpu, tamanhos_entrada
     
def roda_arquivo_especifico():
	escolhido = "01-AchGottundHerr-violin.wav"
	sr, audio = wavfile.read("Representative Dataset/" + escolhido)

	print("Dummy inference")
	predict_tpu(np.zeros(1024), 44100, interpreter, True, True, 10, 1)
	print("\n \n")

	inicio = time.perf_counter()
	get_pitch(audio, sr)
	fim = time.perf_counter()

	print("Levou " + str((fim - inicio)) + " segundos pra rodar na CPU")

	inicio = time.perf_counter()
	get_pitch_tpu(audio, sr, interpreter)
	fim = time.perf_counter()

	print("Levou " + str((fim - inicio)) + " segundos pra rodar na TPU")

	inicio = time.perf_counter()
	freq_yin = librosa.yin(audio.astype("float32"), sr = sr, fmin = 65, fmax = 2500, frame_length = 2822, hop_length = int((sr / 1000) * 10), center = True)
	fim = time.perf_counter()

	print("Levou " + str((fim - inicio)) + " segundos pra rodar com YIN")

	plt.plot(np.linspace(0, len(audio) / sr, len(freq_yin)), freq_yin, color = "green", label = "YIN", alpha = 0.3)

	plt.legend()
	plt.xlabel("Tempo (s)")
	plt.ylabel("Frequência fundamental (Hz)")
	plt.title("Comparação: Predições do modelo completo, quantizado (TPU) e YIN")
	plt.show()
     
media_cpu, media_tpu, tamanhos_entrada = compara_input_aleatorio()
np.savez("compara_performance.npz", media_cpu = media_cpu, media_tpu = media_tpu, tamanhos_entrada = tamanhos_entrada)
