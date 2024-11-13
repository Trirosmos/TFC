import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

compara = np.load("compara_performance.npz")

media_cpu = compara["media_cpu"]
media_tpu = compara["media_tpu"]
tamanhos_entrada = np.array([1024 + ((6144 - 1024)/20) * c for c in range(0, 20)])

duracoes_entrada = tamanhos_entrada / 16000
razao_cpu = media_cpu / duracoes_entrada
razao_tpu = media_tpu / duracoes_entrada

print(duracoes_entrada)
print(media_tpu)
print(media_cpu)
print(razao_tpu)
print(razao_cpu)

ax = plt.gca()

rect = Rectangle((0,0),duracoes_entrada[-1],1,linewidth=1,edgecolor='green',facecolor='green', alpha = 0.3, label = "Tempo real")
rect2 = Rectangle((0,0),0.07,2.33,linewidth=1,edgecolor='orange',facecolor='orange', alpha = 0.3, label = "Latência inicial menor que 70 ms")

ax.add_patch(rect)
ax.add_patch(rect2)


plt.plot(duracoes_entrada, razao_cpu, color = "red", marker = "o", label = "CPU")
plt.plot(duracoes_entrada, razao_tpu, color = "blue", marker = "x", label = "TPU (Modelo quantizado)")
plt.title("Comparação de performance: TPU vs CPU")
plt.xlabel("Duração do sinal de entrada(s)")
plt.ylabel("Razão tempo de processamento/duração do sinal")
plt.legend()
plt.grid()
plt.show()