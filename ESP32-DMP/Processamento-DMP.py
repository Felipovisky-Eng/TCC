# Código para processar o sinal do TCC5.py

import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from scipy.integrate import cumulative_trapezoid
from scipy.fft import rfft, rfftfreq
from mpl_toolkits.mplot3d import Axes3D  # Para plotagem 3D interativa

# --- Configuração visual global ---
plt.rcParams['font.family'] = 'Palatino Linotype'
plt.rcParams['axes.titleweight'] = "bold"
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.labelweight'] = "bold"
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.edgecolor'] = 'gray'
plt.rcParams['grid.color'] = 'lightgray'

# --- 1. Selecionar arquivo CSV ---
Tk().withdraw()
file_path = filedialog.askopenfilename(title="Selecione o arquivo .csv", filetypes=[("CSV files", "*.csv")])
if not file_path:
    print("Nenhum arquivo selecionado.")
    exit()

# --- 2. Ler CSV para np.array ---
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
print(f"{data.shape[0]} amostras carregadas, {data.shape[1]} colunas.")

# --- 3. Separar colunas ---
timestamps_us = data[:,0].astype(np.uint32)
quat = data[:,1:5].astype(np.float32)
accel = data[:,5:8].astype(np.float32)
gyro = data[:,8:11].astype(np.float32)
accel_real = data[:,11:14].astype(np.float32)
accel_world = data[:,14:17].astype(np.float32)
gravity = data[:,17:20].astype(np.float32)
ypr = data[:,20:23].astype(np.float32)

# --- 4. Pré-processamento ---
time_s = (timestamps_us - timestamps_us[0]) / 1e6
n_amostras = len(time_s)

# Escalas físicas (ajuste conforme seu sensor)
ACCEL_SCALE = 9.80665 / 2048  # g -> m/s² e 8G range
GYRO_SCALE = 1.0 / (16.4)         # LSB -> °/s e 1000°/s range

accel_phys = accel * ACCEL_SCALE
gyro_phys = gyro * GYRO_SCALE
accel_real_phys = accel_real * ACCEL_SCALE
accel_world_phys = accel_world * ACCEL_SCALE


# --- 4.5. Corte dos primeiros segundos (amostras iniciais não confiáveis) ---
TEMPO_CORTE = 0  # Cortar os primeiros segundos

# Encontra o índice onde o tempo atinge o valor de corte
indice_corte = np.argmax(time_s >= TEMPO_CORTE)

# Aplica o corte em todos os arrays de dados
time_s = time_s[indice_corte:] - time_s[indice_corte]  # Subtrai o tempo de corte para começar em 0
timestamps_us = timestamps_us[indice_corte:]
quat = quat[indice_corte:,:]
accel = accel[indice_corte:,:]
gyro = gyro[indice_corte:,:]
accel_real = accel_real[indice_corte:,:]
accel_world = accel_world[indice_corte:,:]
gravity = gravity[indice_corte:,:]
ypr = ypr[indice_corte:,:]

# Recalcula variáveis dependentes
n_amostras = len(time_s)
dt = np.diff(time_s)
fs_medio = 1 / np.mean(dt)
tempo_total = time_s[-1] - time_s[0]

print(f"\nPós-corte de {TEMPO_CORTE}s:")
print(f"Novo tempo total: {tempo_total:.3f}s | Amostras restantes: {n_amostras}")
print(f"Fs médio após corte: {fs_medio:.1f}Hz")

# Reaplica as conversões físicas nos dados cortados
accel_phys = accel * ACCEL_SCALE
gyro_phys = gyro * GYRO_SCALE
accel_real_phys = accel_real * ACCEL_SCALE
accel_world_phys = accel_world * ACCEL_SCALE
gyro_rad = np.deg2rad(gyro_phys)
# converter ypr para graus
ypr = np.rad2deg(ypr)

# Reintegra os ângulos YPR com os dados cortados
ypr_integrated = np.zeros_like(gyro_phys, dtype=float)
for i in range(3):
    ypr_integrated[:, i] = cumulative_trapezoid(gyro_phys[:, i], time_s, initial=0)

# --- 5. Estatísticas e jitter ---
dt = np.diff(time_s)
fs_medio = 1 / np.mean(dt)
jitter = dt - np.mean(dt)
jitter_med = np.mean(np.abs(jitter))
jitter_std = np.std(jitter)
tempo_total = time_s[-1] - time_s[0]

print(f"Tempo total: {tempo_total:.3f}s | Amostras: {n_amostras} | Fs médio: {fs_medio:.1f}Hz")
print(f"Jitter médio: {jitter_med*1e3:.3f}ms | Desvio padrão: {jitter_std*1e3:.3f}ms")


# --- 7. Função auxiliar FFT ---
def plot_fft(signal, fs, label, ax=None, use_db=False):
    N = len(signal)
    sig = signal - np.mean(signal)
    fft_vals = rfft(sig)
    freqs = rfftfreq(N, 1/fs)
    magnitude = np.abs(fft_vals)
    if use_db:
        magnitude = 20 * np.log10(magnitude + 1e-12)
    if ax is None:
        plt.figure()
        plt.plot(freqs[1:], magnitude[1:], color=plt.cm.viridis(0.5))
        plt.title(f"FFT {label}")
        plt.xlabel("Frequência (Hz)")
        plt.ylabel("Magnitude" + (" [dB]" if use_db else ""))
        plt.grid(True, linestyle='--', alpha=0.6)
    else:
        ax.plot(freqs[1:], magnitude[1:], label=label, color=plt.cm.viridis(0.5))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

# --- 8. Plot jitter e tempo ---
plt.figure()
plt.subplot(2,1,1)
plt.plot(dt*1e3, marker='.', markersize=2)
plt.title("Jitter entre amostras")
plt.ylabel("Δt [ms]")
plt.grid(True, linestyle='--', alpha=0.7)
plt.subplot(2,1,2)
plt.plot(time_s, marker='.', markersize=2)
plt.title("Tempo acumulado")
plt.xlabel("Amostra")
plt.ylabel("Tempo [s]")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 9. Plot aceleração ---
fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
for i, axis in enumerate(['X','Y','Z']):
    ax[i].plot(time_s, accel_phys[:,i], label=f"Aceleração {axis}", color=plt.cm.viridis(0.2 + i*0.3))
    ax[i].set_ylabel("m/s²")
    ax[i].grid(True, linestyle='--', alpha=0.6)
    ax[i].legend()
ax[-1].set_xlabel("Tempo [s]")
fig.suptitle("Aceleração Física")
plt.tight_layout()
plt.show()

# --- 10. Plot giroscópio ---
fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
for i, axis in enumerate(['X','Y','Z']):
    ax[i].plot(time_s, gyro_phys[:,i], label=f"Giroscópio {axis}", color=plt.cm.viridis(0.2 + i*0.3))
    ax[i].set_ylabel("°/s")
    ax[i].grid(True, linestyle='--', alpha=0.6)
    ax[i].legend()
ax[-1].set_xlabel("Tempo [s]")
fig.suptitle("Velocidade Angular Física")
plt.tight_layout()
plt.show()

# --- 11. Plot aceleração real ---
fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
for i, axis in enumerate(['X','Y','Z']):
    ax[i].plot(time_s, accel_real_phys[:,i], label=f"Aceleração Real {axis}", color=plt.cm.viridis(0.2 + i*0.3))
    ax[i].set_ylabel("m/s²")
    ax[i].grid(True, linestyle='--', alpha=0.6)
    ax[i].legend()
ax[-1].set_xlabel("Tempo [s]")
fig.suptitle("Aceleração Real (sem gravidade)")
plt.tight_layout()
plt.show()

# --- 12. Plot aceleração world ---
fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
for i, axis in enumerate(['X','Y','Z']):
    ax[i].plot(time_s, accel_world_phys[:,i], label=f"Aceleração World {axis}", color=plt.cm.viridis(0.2 + i*0.3))
    ax[i].set_ylabel("m/s²")
    ax[i].grid(True, linestyle='--', alpha=0.6)
    ax[i].legend()
ax[-1].set_xlabel("Tempo [s]")
fig.suptitle("Aceleração no Referencial do Mundo")
plt.tight_layout()
plt.show()

# --- 13. Plot YPR vs Integração ---
fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
for i, axis in enumerate(['Yaw','Pitch','Roll']):
    ax[i].plot(time_s, ypr[:,i], label=f"YPR {axis}", color=plt.cm.viridis(0.2))
    ax[i].plot(time_s, (ypr_integrated[:,i]), '--', label=f"Integrado {axis}", color=plt.cm.viridis(0.7))
    ax[i].set_ylabel("°")
    ax[i].grid(True, linestyle='--', alpha=0.6)
    ax[i].legend()
ax[-1].set_xlabel("Tempo [s]")
fig.suptitle("YPR vs Integração de Giroscópio")
plt.tight_layout()
plt.show()

# --- 13.5 Plot YPR apenas ---
fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
for i, axis in enumerate(['Yaw','Pitch','Roll']):
    ax[i].plot(time_s, ypr[:,i], label=f"YPR {axis}", color=plt.cm.viridis(0.2 + i*0.3))
    ax[i].set_ylabel("°")
    ax[i].grid(True, linestyle='--', alpha=0.6)
    ax[i].legend()
ax[-1].set_xlabel("Tempo [s]")
fig.suptitle("YPR")
plt.tight_layout()
plt.show()


# --- 15. FFT do acelerômetro ---
fig, axs = plt.subplots(3,1, figsize=(8,6), sharex=True)
for i, axis in enumerate(['X','Y','Z']):
    plot_fft(accel_phys[:,i], fs_medio, f"Acel {axis}", ax=axs[i])
    axs[i].set_ylabel("Magnitude")
axs[-1].set_xlabel("Frequência [Hz]")
fig.suptitle("FFT do Acelerômetro")
plt.tight_layout()
plt.show()

# --- 16. FFT do giroscópio ---
fig, axs = plt.subplots(3,1, figsize=(8,6), sharex=True)
for i, axis in enumerate(['X','Y','Z']):
    plot_fft(gyro_phys[:,i], fs_medio, f"Giro {axis}", ax=axs[i])
    axs[i].set_ylabel("Magnitude")
axs[-1].set_xlabel("Frequência [Hz]")
fig.suptitle("FFT do Giroscópio")
plt.tight_layout()
plt.show()


# --- 18. FFT do acelerômetro (linear e dB) ---
from scipy.fft import rfft, rfftfreq

# Função para plot FFT
def plot_fft(signal, fs, axis_label, tipo, use_db=False, ax=None):
    N = len(signal)
    signal_zero_mean = signal - np.mean(signal)
    fft_vals = rfft(signal_zero_mean)
    freqs = rfftfreq(N, 1/fs)
    magnitude = np.abs(fft_vals[1:])  # remove DC
    if use_db:
        magnitude = 20 * np.log10(magnitude + 1e-12)
    freqs_no_dc = freqs[1:]

    color = plt.cm.viridis(0.4)
    if ax is None:
        plt.plot(freqs_no_dc, magnitude, color=color, label=f"{tipo} {axis_label}")
        plt.title(f"FFT do {tipo} eixo {axis_label}" + (" (dB)" if use_db else ""))
        plt.xlabel("Frequência [Hz]")
        plt.ylabel("Magnitude [dB]" if use_db else "Magnitude")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=8)
    else:
        ax.plot(freqs_no_dc, magnitude, color=color, label=f"{tipo} {axis_label}", linewidth=1)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8)

# --- 18.2 FFT em dB do acelerômetro ---
fig, axs = plt.subplots(3,1, figsize=(10,8), sharex=True)
for i, axis in enumerate(['X','Y','Z']):
    plot_fft(accel_phys[:,i], fs_medio, axis, "Acel", use_db=True, ax=axs[i])
    axs[i].set_ylabel("Magnitude [dB]")
axs[-1].set_xlabel("Frequência [Hz]")
fig.suptitle("FFT do Acelerômetro (Escala dB)", fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

# --- 19.2 FFT em dB do giroscópio ---
fig, axs = plt.subplots(3,1, figsize=(10,8), sharex=True)
for i, axis in enumerate(['X','Y','Z']):
    plot_fft(gyro_phys[:,i], fs_medio, axis, "Giro", use_db=True, ax=axs[i])
    axs[i].set_ylabel("Magnitude [dB]")
axs[-1].set_xlabel("Frequência [Hz]")
fig.suptitle("FFT do Giroscópio (Escala dB)", fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

# --- 20. Integração da aceleração world para obter velocidade --- 
# Converte aceleração para m/s² (já está em accel_world_phys) e integra usando trapézio
vel_world = cumulative_trapezoid(accel_world_phys, time_s, axis=0, initial=0)

# Estatísticas básicas da velocidade
#print_stats("Velocidade (m/s)", vel_world)

# Plot da velocidade world (3 eixos)
fig, ax = plt.subplots(figsize=(7,3.5))
for i, axis in enumerate(['X','Y','Z']):
    ax.plot(time_s, vel_world[:,i], label=f"Velocidade {axis}", color=plt.cm.viridis(0.2 + i*0.3))
ax.set_title("Velocidade no Referencial do Mundo (m/s)")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Velocidade (m/s)")
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()


# --- 21. Integração da velocidade world para obter deslocamento --- 
# Integra velocidade para obter posição/deslocamento
disp_world = cumulative_trapezoid(vel_world, time_s, axis=0, initial=0)

# Estatísticas básicas do deslocamento
#print_stats("Deslocamento (m)", disp_world)


# --- 22. Visualização do deslocamento 3D, projeções 2D e métricas ---

from mpl_toolkits.mplot3d import Axes3D  # ok mesmo sem uso direto

# --- 22.1. Métricas ---
# Deslocamento acumulado (módulo do vetor posição ao longo do tempo)
disp_accum = np.linalg.norm(disp_world, axis=1)

# Comprimento de arco (distância percorrida real ao longo da trajetória)
step_distances = np.linalg.norm(np.diff(disp_world, axis=0), axis=1)
arc_length = np.concatenate(([0.0], np.cumsum(step_distances)))

print("\n--- Métricas de deslocamento ---")
print(f"Deslocamento final (reta origem→fim): {disp_accum[-1]:.3f} m")
print(f"Distância percorrida (comprimento de arco): {arc_length[-1]:.3f} m")


# --- 22.4. Deslocamento acumulado e distância percorrida vs tempo ---
plt.figure(figsize=(10,6))
plt.plot(time_s, disp_accum, label="Deslocamento acumulado (reta)")
plt.plot(time_s, arc_length, label="Distância percorrida (arco)")
plt.xlabel("Tempo [s]")
plt.ylabel("Distância [m]")
plt.title("Evolução: deslocamento e distância percorrida")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# --- 23. Aplicação de filtros digitais ---
from scipy.signal import butter, filtfilt

# Configurações dos filtros
FS = fs_medio  # Frequência de amostragem (já calculada)
CUTOFF_LOW = 0.1  # Frequência de corte para passa-alta (Hz)
CUTOFF_HIGH = 10.0  # Frequência de corte para passa-baixa (Hz)
ORDER = 4     # Ordem do filtro

# Funções auxiliares para criação de filtros 
def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_filter(data, cutoff, fs, filter_type, order=4):
    """Aplica filtro passa-alta ou passa-baixa com fase zero"""
    if filter_type == 'high':
        b, a = butter_highpass(cutoff, fs, order=order)
    elif filter_type == 'low':
        b, a = butter_lowpass(cutoff, fs, order=order)
    
    return filtfilt(b, a, data, axis=0)

# --- 23.1 Filtro passa-alta na velocidade angular (giroscópio) ---
gyro_phys_filtrado = np.zeros_like(gyro_phys)
for i in range(3):
    gyro_phys_filtrado[:,i] = apply_filter(gyro_phys[:,i], CUTOFF_LOW, FS, 'high')

# Reintegra com os dados filtrados
gyro_rad_filtrado = np.deg2rad(gyro_phys_filtrado)
ypr_integrado_filtrado = np.zeros_like(gyro_rad_filtrado)
for i in range(3):
    ypr_integrado_filtrado[:,i] = cumulative_trapezoid(gyro_rad_filtrado[:,i], time_s, initial=0)

# --- 23.2 Filtro passa-alta na velocidade obtida da aceleração world ---
vel_world_filtrada = np.zeros_like(vel_world)
for i in range(3):
    vel_world_filtrada[:,i] = apply_filter(vel_world[:,i], CUTOFF_LOW, FS, 'high')

# --- 23.3 Visualização comparativa ---
# Comparação YPR original vs integrado filtrado
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
titles = ['Yaw (Z)', 'Pitch (Y)', 'Roll (X)']
for i in range(3):
    axs[i].plot(time_s, ypr[:,i], label='YPR (MPU)', alpha=0.7)
    axs[i].plot(time_s, np.rad2deg(ypr_integrated[:,i]), '--', label='Giroscópio integrado (original)', alpha=0.7)
    axs[i].plot(time_s, np.rad2deg(ypr_integrado_filtrado[:,i]), label='Giroscópio integrado (filtrado HP)', linewidth=1.5)
    axs[i].set_title(titles[i])
    axs[i].set_ylabel('Ângulo (°)')
    axs[i].grid(True, linestyle='--', alpha=0.6)
    axs[i].legend(fontsize=8)
axs[-1].set_xlabel('Tempo (s)')
fig.suptitle('Comparação: YPR vs Giroscópio Integrado (com e sem filtro passa-alta)', fontweight='bold')
plt.tight_layout()
plt.show()

# Comparação Velocidade world original vs filtrada
fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
for i, axis in enumerate(['X', 'Y', 'Z']):
    axs[i].plot(time_s, vel_world[:,i], label=f'Velocidade {axis} (original)', alpha=0.7)
    axs[i].plot(time_s, vel_world_filtrada[:,i], label=f'Velocidade {axis} (filtrada HP)', linewidth=1.5)
    axs[i].set_ylabel('Velocidade (m/s)')
    axs[i].grid(True, linestyle='--', alpha=0.6)
    axs[i].legend(fontsize=8)
axs[-1].set_xlabel('Tempo (s)')
fig.suptitle('Velocidade World: Comparação antes e depois do filtro passa-alta', fontweight='bold')
plt.tight_layout()
plt.show()

# --- 23.4 Atualização das variáveis para uso posterior ---
# (Opcional) Descomente se quiser substituir as variáveis originais pelos dados filtrados
# vel_world = vel_world_filtrada
# ypr_integrated = ypr_integrado_filtrado

# --- 23.5 Integração da velocidade filtrada para obter deslocamento ---

# Integra a velocidade filtrada para obter novo deslocamento
disp_world_filtrado = cumulative_trapezoid(vel_world_filtrada, time_s, axis=0, initial=0)

# Calcula métricas do deslocamento filtrado
disp_accum_filtrado = np.linalg.norm(disp_world_filtrado, axis=1)
step_distances_filtrado = np.linalg.norm(np.diff(disp_world_filtrado, axis=0), axis=1)
arc_length_filtrado = np.concatenate(([0.0], np.cumsum(step_distances_filtrado)))

print("\n--- Métricas de deslocamento com velocidade filtrada ---")
print(f"Deslocamento final (filtrado): {disp_accum_filtrado[-1]:.3f} m")
print(f"Distância percorrida (filtrado): {arc_length_filtrado[-1]:.3f} m")

# --- Visualização comparativa: deslocamento original vs filtrado ---

# Gráfico 1: Comparação das trajetórias 3D
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(disp_world[:,0], disp_world[:,1], disp_world[:,2], 
         label='Original', linewidth=1.5, alpha=0.7)
ax1.set_title("Deslocamento Original")
ax1.set_xlabel("X [m]"); ax1.set_ylabel("Y [m]"); ax1.set_zlabel("Z [m]")
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(disp_world_filtrado[:,0], disp_world_filtrado[:,1], disp_world_filtrado[:,2],
         color='orange', label='Filtrado', linewidth=1.5)
ax2.set_title("Deslocamento com Velocidade Filtrada")
ax2.set_xlabel("X [m]"); ax2.set_ylabel("Y [m]"); ax2.set_zlabel("Z [m]")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Gráfico 2: Componentes do deslocamento por eixo
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, axis in enumerate(['X', 'Y', 'Z']):
    axs[i].plot(time_s, disp_world[:,i], label='Original', alpha=0.7)
    axs[i].plot(time_s, disp_world_filtrado[:,i], label='Filtrado', linewidth=1.5)
    axs[i].set_ylabel(f"Desloc. {axis} [m]")
    axs[i].grid(True, linestyle='--', alpha=0.6)
    axs[i].legend(fontsize=8)
axs[-1].set_xlabel("Tempo [s]")
fig.suptitle("Comparação: Deslocamento nos Eixos - Original vs Filtrado", fontweight='bold')
plt.tight_layout()

# Gráfico 3: Deslocamento acumulado e distância percorrida
plt.figure(figsize=(10, 6))
plt.plot(time_s, disp_accum, label="Desloc. acumulado (original)")
plt.plot(time_s, arc_length, label="Distância percorrida (original)")
plt.plot(time_s, disp_accum_filtrado, '--', label="Desloc. acumulado (filtrado)")
plt.plot(time_s, arc_length_filtrado, '--', label="Distância percorrida (filtrado)")
plt.xlabel("Tempo [s]")
plt.ylabel("Distância [m]")
plt.title("Evolução: Deslocamento e Distância - Original vs Filtrado")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

# --- Atualização opcional das variáveis ---
# Descomente para usar os dados filtrados nas análises posteriores
# disp_world = disp_world_filtrado
# disp_accum = disp_accum_filtrado
# arc_length = arc_length_filtrado