#Código preliminar para o TCC (Sim, tiveram 10 antes desse, um pouca mias na realidade)

"""
Módulo 1 + 2: Seleção de arquivo CSV e pré-processamento
- Seleciona arquivo CSV via diálogo
- Lê os dados para NumPy arrays
- Normaliza tempo e altitude
- Converte aceleração para m/s²
- Aplica offsets e escalas
- Calcula estatísticas iniciais
"""

import numpy as np
import struct
from tkinter import Tk, filedialog
import os

# ================================
# 1. Seleção do arquivo CSV
# ================================
def selecionar_arquivo_csv():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecione o arquivo .csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        print("Nenhum arquivo selecionado.")
        exit()
    return file_path

# ================================
# 2. Leitura CSV para NumPy arrays
# ================================
def ler_csv_para_numpy(file_path):
    """
    Lê CSV e retorna arrays NumPy separados
    """
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    print(f"{data.shape[0]} amostras carregadas, {data.shape[1]} colunas.")

    timestamps_us = data[:,0].astype(np.uint32)
    mpu_accel = data[:,1:4].astype(np.float32)
    mpu_gyro  = data[:,4:7].astype(np.float32)
    adxl_accel = data[:,7:10].astype(np.float32)
    adxl_accel = adxl_accel[:, [1, 0, 2]]  # troca X↔Y
    altitude = data[:,10].astype(np.float32)

    return timestamps_us, mpu_accel, mpu_gyro, adxl_accel, altitude

file_path = selecionar_arquivo_csv()
timestamps_us, mpu_accel, mpu_gyro, adxl_accel, altitude = ler_csv_para_numpy(file_path)

# ================================
# 3. Pré-processamento
# ================================
def preprocess_sensor_data(timestamps_us, mpu_accel, mpu_gyro, adxl_accel, altitude):
    """
    - Normaliza tempo (t=0)
    - Calcula média dos sensores
    - Normaliza altitude
    - Aplica offsets e escalas
    - Converte aceleração para m/s²
    """
    # --- Tempo normalizado ---
    t_s = (timestamps_us - timestamps_us[0]) / 1e6  # micros → s

    # --- Estatísticas iniciais ---
    N = len(t_s)
    Fs = N / (t_s[-1]-t_s[0]) if N > 1 else 0
    T_total = t_s[-1] - t_s[0]

    #Calculo de frequênia média

    FS = 1 / np.mean(np.diff(t_s)) if N > 1 else 0

    print(f"\nNúmero de amostras: {N}")
    print(f"Frequência média (Hz): {Fs:.2f}")
    print(f"Frequência média (Hz): {FS:.2f}")
    print(f"Duração total (s): {T_total:.3f}")

    # --- Normaliza altitude ---
    altitude_proc = altitude - altitude[0]

    # --- Escalas e offsets ---
    # Constantes do MPU e ADXL
    g = 9.80665  # m/s²
    MPU_ACCEL_SCALE = 8/32768 * g       # ±8g convertido para m/s²
    MPU_GYRO_SCALE  = 1000/32768        # ±1000º/s
    ADXL_SCALE      = 1/20.5 * g      # Exemplo: cada LSB = 0.2g → m/s²

    # Offsets (defina conforme calibração manual)
    mpu_accel_offset = np.array([-360,-21,-86], dtype=np.float32)
    mpu_gyro_offset  = np.array([-47.59,38.8,15.8], dtype=np.float32)
    adxl_accel_offset = np.array([-4.6,1.8,4.0], dtype=np.float32)

    # --- Aplicar offsets e escalas ---
    mpu_accel_proc = (mpu_accel + mpu_accel_offset) * MPU_ACCEL_SCALE
    mpu_gyro_proc  = (mpu_gyro  + mpu_gyro_offset)  * MPU_GYRO_SCALE
    adxl_accel_proc = (adxl_accel + adxl_accel_offset) * ADXL_SCALE

    # --- Médias ---

        # --- Médias ---
    print("\nMédias dos sensores não convertidos (primeiras análises):")
    print(f"MPU Accel (m/s²): {mpu_accel.mean(axis=0)}")
    print(f"MPU Gyro (º/s)  : {mpu_gyro.mean(axis=0)}")
    print(f"ADXL Accel (m/s²): {adxl_accel.mean(axis=0)}")
    print(f"Altitude (m)   : {altitude.mean():.3f}")


    print("\nMédias dos sensores convertidos (primeiras análises):")
    print(f"MPU Accel (m/s²): {mpu_accel_proc.mean(axis=0)}")
    print(f"MPU Gyro (º/s)  : {mpu_gyro_proc.mean(axis=0)}")
    print(f"ADXL Accel (m/s²): {adxl_accel_proc.mean(axis=0)}")
    print(f"Altitude (m)   : {altitude_proc.mean():.3f}")

    return t_s, mpu_accel_proc, mpu_gyro_proc, adxl_accel_proc, altitude_proc, FS

t_s, mpu_accel_proc, mpu_gyro_proc, adxl_accel_proc, altitude_proc, FS = preprocess_sensor_data(timestamps_us, mpu_accel, mpu_gyro, adxl_accel, altitude)

# ======================== Módulo 4: Pré-exibição / Plots ========================
import matplotlib.pyplot as plt

# --- Configuração visual global ---
plt.rcParams['font.family'] = 'Palatino Linotype'
plt.rcParams['axes.titleweight'] = "bold"
plt.rcParams['figure.dpi'] = 160
plt.rcParams['axes.labelweight'] = "bold"
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.edgecolor'] = 'gray'
plt.rcParams['grid.color'] = 'lightgray'

# ======================== Funções de plot ========================

def plot_accel_mpu(time_s, mpu_accel):
    fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
    for i, axis in enumerate(['X','Y','Z']):
        ax[i].plot(time_s, mpu_accel[:,i], label=f"Aceleração MPU {axis}", color=plt.cm.viridis(0.2 + i*0.3))
        ax[i].set_ylabel("m/s²")
        ax[i].grid(True, linestyle='--', alpha=0.6)
        ax[i].legend()
    ax[-1].set_xlabel("Tempo [s]")
    fig.suptitle("Aceleração MPU")
    plt.tight_layout()
    plt.show()

def plot_gyro_mpu(time_s, mpu_gyro):
    fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
    for i, axis in enumerate(['X','Y','Z']):
        ax[i].plot(time_s, mpu_gyro[:,i], label=f"Giroscópio MPU {axis}", color=plt.cm.viridis(0.2 + i*0.3))
        ax[i].set_ylabel("°/s")
        ax[i].grid(True, linestyle='--', alpha=0.6)
        ax[i].legend()
    ax[-1].set_xlabel("Tempo [s]")
    fig.suptitle("Velocidade Angular MPU")
    plt.tight_layout()
    plt.show()

def plot_accel_adxl(time_s, adxl_accel):
    fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
    for i, axis in enumerate(['X','Y','Z']):
        ax[i].plot(time_s, adxl_accel[:,i], label=f"Aceleração ADXL {axis}", color=plt.cm.viridis(0.2 + i*0.3))
        ax[i].set_ylabel("m/s²")
        ax[i].grid(True, linestyle='--', alpha=0.6)
        ax[i].legend()
    ax[-1].set_xlabel("Tempo [s]")
    fig.suptitle("Aceleração ADXL")
    plt.tight_layout()
    plt.show()

def plot_altitude_bmp(time_s, altitude):
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    ax.plot(time_s, altitude, label="Altitude BMP", color=plt.cm.viridis(0.5))
    ax.set_ylabel("Altitude [m]")
    ax.set_xlabel("Tempo [s]")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.suptitle("Altitude BMP")
    plt.tight_layout()
    plt.show()

def plot_jitter_tempo(time_s):
    # Calcular jitter
    dt = np.diff(time_s)
    jitter_ms = dt * 1000  # converter para ms

    fig, ax = plt.subplots(2,1, figsize=(8,5), sharex=False)
    
    # Subplot jitter
    ax[0].plot(jitter_ms, label="Jitter entre amostras", color=plt.cm.viridis(0.5))
    ax[0].set_ylabel("Δt [ms]")
    ax[0].grid(True, linestyle='--', alpha=0.6)
    ax[0].legend()

    # Subplot tempo em função do índice da amostra
    ax[1].plot(time_s, label="Tempo acumulado", color=plt.cm.viridis(0.7))
    ax[1].set_xlabel("Amostra")
    ax[1].set_ylabel("Tempo [s]")
    ax[1].grid(True, linestyle='--', alpha=0.6)
    ax[1].legend()

    fig.suptitle("Jitter e Tempo por Amostra")
    plt.tight_layout()
    plt.show()

# ======================== Função principal de pré-exibição ========================
def pre_exibicao(time_s, mpu_accel, mpu_gyro, adxl_accel, altitude):
    """
    Executa todos os plots de pré-exibição:
    - Aceleração MPU
    - Giroscópio MPU
    - Aceleração ADXL
    - Altitude BMP
    - Jitter e tempo por amostra
    """
    plot_accel_mpu(time_s, mpu_accel)
    plot_gyro_mpu(time_s, mpu_gyro)
    plot_accel_adxl(time_s, adxl_accel)
    plot_altitude_bmp(time_s, altitude)
    plot_jitter_tempo(time_s)

pre_exibicao(t_s, mpu_accel_proc, mpu_gyro_proc, adxl_accel_proc, altitude_proc)

# ======================== Módulo 5: Pré-análise de frequência (FFT) ========================
from scipy.fft import rfft, rfftfreq

# --- Função genérica para plot FFT ---
def plot_fft(signal, fs, axis_label, tipo, ax=None):
    """
    Plota a FFT de um sinal removendo o componente DC
    signal: array do sinal (1D)
    fs: frequência de amostragem [Hz]
    axis_label: 'X', 'Y', 'Z' ou 'Altitude'
    tipo: 'MPU Accel', 'MPU Gyro', 'ADXL Accel', 'BMP Altitude'
    ax: eixo de plot opcional (subplots)
    """
    N = len(signal)
    signal_zero_mean = signal - np.mean(signal)  # remove DC
    fft_vals = rfft(signal_zero_mean)
    freqs = rfftfreq(N, 1/fs)
    
    magnitude = np.abs(fft_vals[1:])  # remove componente DC
    freqs_no_dc = freqs[1:]

    color = plt.cm.viridis(0.4)
    
    if ax is None:
        plt.figure(figsize=(8,4))
        plt.plot(freqs_no_dc, magnitude, color=color, label=f"{tipo} eixo {axis_label}")
        plt.title(f"FFT do {tipo} eixo {axis_label}")
        plt.xlabel("Frequência [Hz]")
        plt.ylabel("Magnitude")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        ax.plot(freqs_no_dc, magnitude, color=color, label=f"{tipo} eixo {axis_label}", linewidth=1)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8)

# ======================== Funções específicas por sensor ========================

def fft_accel_mpu(time_s, mpu_accel):
    fs = 1 / np.mean(np.diff(time_s))
    fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
    for i, axis in enumerate(['X','Y','Z']):
        plot_fft(mpu_accel[:,i], fs, axis, "MPU Accel", ax[i])
        ax[i].set_ylabel("Magnitude [m/s²]")
    ax[-1].set_xlabel("Frequência [Hz]")
    fig.suptitle("FFT Aceleração MPU")
    plt.tight_layout()
    plt.show()

def fft_gyro_mpu(time_s, mpu_gyro):
    fs = 1 / np.mean(np.diff(time_s))
    fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
    for i, axis in enumerate(['X','Y','Z']):
        plot_fft(mpu_gyro[:,i], fs, axis, "MPU Gyro", ax[i])
        ax[i].set_ylabel("Magnitude [°/s]")
    ax[-1].set_xlabel("Frequência [Hz]")
    fig.suptitle("FFT Velocidade Angular MPU")
    plt.tight_layout()
    plt.show()

def fft_accel_adxl(time_s, adxl_accel):
    fs = 1 / np.mean(np.diff(time_s))
    fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
    for i, axis in enumerate(['X','Y','Z']):
        plot_fft(adxl_accel[:,i], fs, axis, "ADXL Accel", ax[i])
        ax[i].set_ylabel("Magnitude [m/s²]")
    ax[-1].set_xlabel("Frequência [Hz]")
    fig.suptitle("FFT Aceleração ADXL")
    plt.tight_layout()
    plt.show()

def fft_altitude_bmp(time_s, altitude):
    fs = 1 / np.mean(np.diff(time_s))
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    plot_fft(altitude, fs, "Altitude", "BMP Altitude", ax)
    ax.set_ylabel("Magnitude [m]")
    ax.set_xlabel("Frequência [Hz]")
    fig.suptitle("FFT Altitude BMP")
    plt.tight_layout()
    plt.show()

# ======================== Função principal do módulo ========================
def pre_analise_frequencia(time_s, mpu_accel, mpu_gyro, adxl_accel, altitude):
    """
    Executa FFT de todos os sinais e plota:
    - MPU aceleração
    - MPU giroscópio
    - ADXL aceleração
    - Altitude BMP
    """
    fft_accel_mpu(time_s, mpu_accel)
    fft_gyro_mpu(time_s, mpu_gyro)
    fft_accel_adxl(time_s, adxl_accel)
    fft_altitude_bmp(time_s, altitude)

pre_analise_frequencia(t_s, mpu_accel_proc, mpu_gyro_proc, adxl_accel_proc, altitude_proc)

# ======================== Módulo 6: Interpolação cúbica MPU e ADXL ========================
from scipy.interpolate import interp1d

def interpolar_dados(time_s, mpu_accel, mpu_gyro, adxl_accel, FS, fator_interpolacao=20, ativar=True):
    """
    Interpola os dados do MPU e ADXL aumentando o número de amostras
    e regularizando o vetor de tempo.

    Parâmetros:
    ------------
    time_s : ndarray
        Vetor de tempo original (s)
    mpu_accel : ndarray (N,3)
        Aceleração MPU
    mpu_gyro : ndarray (N,3)
        Velocidade angular MPU
    adxl_accel : ndarray (N,3)
        Aceleração ADXL
    FS : float
        Frequência de amostragem original
    fator_interpolacao : int
        Fator de aumento de amostragem
    ativar : bool
        Se True, aplica interpolação; se False, retorna os dados originais

    Retorna:
    --------
    time_s_interp : ndarray
        Vetor de tempo interpolado
    mpu_accel_interp : ndarray (N*fator,3)
        Aceleração MPU interpolada
    mpu_gyro_interp : ndarray (N*fator,3)
        Velocidade angular MPU interpolada
    adxl_accel_interp : ndarray (N*fator,3)
        Aceleração ADXL interpolada
    FS_novo : float
        Nova frequência de amostragem
    """

    if not ativar:
        print("Interpolação desativada — usando dados originais.")
        return time_s, mpu_accel, mpu_gyro, adxl_accel, FS

    # Número de pontos interpolados
    N_interp = len(time_s) * fator_interpolacao
    time_s_interp = np.linspace(time_s[0], time_s[-1], N_interp)

    # Função de interpolação cúbica por array (axis=0)
    interp_accel = interp1d(time_s, mpu_accel, kind='cubic', axis=0, fill_value='extrapolate')
    interp_gyro = interp1d(time_s, mpu_gyro, kind='cubic', axis=0, fill_value='extrapolate')
    interp_adxl = interp1d(time_s, adxl_accel, kind='cubic', axis=0, fill_value='extrapolate')

    # Aplicar interpolação
    mpu_accel_interp = interp_accel(time_s_interp)
    mpu_gyro_interp  = interp_gyro(time_s_interp)
    adxl_accel_interp = interp_adxl(time_s_interp)

    # Nova frequência de amostragem
    FS_novo = FS * fator_interpolacao

    print(f"Interpolação ativada — {fator_interpolacao}x mais amostras.")
    print(f"Nova frequência de amostragem: {FS_novo:.2f} Hz")

    # --- Gráficos comparativos ---
    # Configuração visual global já aplicada anteriormente

    sensores = {
        "MPU Aceleração": mpu_accel_interp,
        "MPU Giroscópio": mpu_gyro_interp,
        "ADXL Aceleração": adxl_accel_interp
    }
    originais = {
        "MPU Aceleração": mpu_accel,
        "MPU Giroscópio": mpu_gyro,
        "ADXL Aceleração": adxl_accel
    }

    for nome, dados_interp in sensores.items():
        fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
        for i, eixo in enumerate(['X','Y','Z']):
            ax[i].plot(time_s_interp, dados_interp[:,i], '-', label=f'Interpolado {eixo}', color=plt.cm.viridis(0.2 + i*0.3))
            ax[i].plot(time_s, originais[nome][:,i], 'o', markersize=3, alpha=0.6, label=f'Original {eixo}', color=plt.cm.viridis(0.2 + i*0.3))
            ax[i].set_ylabel("m/s²" if "Aceleração" in nome else "°/s")
            ax[i].grid(True, linestyle='--', alpha=0.6)
            ax[i].legend()
        ax[-1].set_xlabel("Tempo [s]")
        fig.suptitle(f"{nome} — Comparação Original vs Interpolado")
        plt.tight_layout()
        plt.show()

    return time_s_interp, mpu_accel_interp, mpu_gyro_interp, adxl_accel_interp, FS_novo

time_s_interp, mpu_accel_interp, mpu_gyro_interp, adxl_accel_interp, FS_novo = interpolar_dados(
    t_s, mpu_accel_proc, mpu_gyro_proc, adxl_accel_proc, FS,
    fator_interpolacao=20,
    ativar=True
)

# ================= MÓDULO 7 – FILTRAGEM =================
from scipy.signal import butter, filtfilt, firwin

def filtrar_dados(time_s, FS, FS_novo, mpu_accel, mpu_gyro, adxl_accel, altitude_proc, ativar=True):
    """
    Aplica filtros nos dados do MPU e ADXL:
    - IIR Butter PB 2ª ordem, fc = 25 Hz (todos)
    - FIR PA 0,5 Hz ordem 101 (apenas MPU gyro)
    """

    if not ativar:
        print("Filtragem desativada — retornando dados originais")
        return mpu_accel, mpu_gyro, adxl_accel

    # --- IIR Butter PB 25Hz --- MPU e ADXL ---
    fc_pb = 15
    nyq = 0.5 * FS_novo
    Wn = fc_pb / nyq
    b, a = butter(2, Wn, btype='low')

    # --- IIR Butter PB 15Hz --- BMP ---
    fc_pb = 15
    nyq_bmp = 0.5 * FS
    Wn_bmp = fc_pb / nyq_bmp
    b_bmp, a_bmp = butter(4, Wn_bmp, btype='low')


    mpu_accel_filt = np.zeros_like(mpu_accel)
    mpu_gyro_filt  = np.zeros_like(mpu_gyro)
    adxl_accel_filt = np.zeros_like(adxl_accel)
    altitude_filt   = np.zeros_like(altitude_proc)

    #Filtragem IIR
    for i in range(3):
        mpu_accel_filt[:,i] = filtfilt(b, a, mpu_accel[:,i])
        mpu_gyro_filt[:,i]  = filtfilt(b, a, mpu_gyro[:,i])
        adxl_accel_filt[:,i] = filtfilt(b, a, adxl_accel[:,i])
    
    altitude_filt = filtfilt(b_bmp, a_bmp, altitude_proc)

    # --- FIR PA 0,5Hz para giroscópio MPU ---
    fc_pa = 0.5
    ordem_fir = 901
    Wn_pa = fc_pa / nyq
    fir_coeff = firwin(ordem_fir, Wn_pa, pass_zero=False)
    for i in range(3):
        mpu_gyro_filt[:,i] = filtfilt(fir_coeff, [1.0], mpu_gyro_filt[:,i])

    # --- Gráficos comparativos ---
    sensores = {'MPU Accel [m/s²]': mpu_accel, 
                'MPU Gyro [°/s]': mpu_gyro, 
                'ADXL Accel [m/s²]': adxl_accel}

    sensores_filt = {'MPU Accel [m/s²]': mpu_accel_filt, 
                     'MPU Gyro [°/s]': mpu_gyro_filt, 
                     'ADXL Accel [m/s²]': adxl_accel_filt}

    for nome, dados in sensores.items():
        fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
        for i, eixo in enumerate(['X','Y','Z']):
            ax[i].plot(time_s, dados[:,i], 'o', label=f"{nome} Original {eixo}", alpha=0.5, markersize=3)
            ax[i].plot(time_s, sensores_filt[nome][:,i], '-', label=f"{nome} Filtrado {eixo}")
            ax[i].set_ylabel(nome.split()[1])
            ax[i].grid(True, linestyle='--', alpha=0.6)
            ax[i].legend(fontsize=8)
        ax[-1].set_xlabel("Tempo [s]")
        fig.suptitle(f"{nome} – Original x Filtrado")
        plt.tight_layout()
        plt.show()

    return mpu_accel_filt, mpu_gyro_filt, adxl_accel_filt, altitude_filt

mpu_accel_filt, mpu_gyro_filt, adxl_accel_filt, altitude_filt = filtrar_dados(
    time_s_interp, FS, FS_novo,
    mpu_accel_interp, mpu_gyro_interp, adxl_accel_interp, altitude_proc,
    ativar=True
)

# ================= MÓDULO 7.1 – INTERPOLAÇÃO BMP =================

def interpolar_bmp(time_orig, bmp_proc, bmp_alt, time_interp, ativar=True):

    #Compara BMP com e sem filtro

    plt.figure(figsize=(8,4))
    plt.plot(time_orig, bmp_proc, 'o', alpha=0.5, label='BMP Original')
    plt.plot(time_orig, bmp_alt, '-', label='BMP Filtrado')
    plt.xlabel("Tempo [s]")
    plt.ylabel("Altitude [m]")
    plt.title("Altitude BMP")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    """
    Interpola dados do BMP com vetor de tempo interpolado dos sensores.
    """
    if not ativar:
        print("Interpolação do BMP desativada — retornando dados originais")
        return bmp_alt

    interp_func = interp1d(time_orig, bmp_alt, kind='cubic', fill_value="extrapolate")
    bmp_alt_interp = interp_func(time_interp)

    # --- Gráfico comparativo ---
    plt.figure(figsize=(8,4))
    plt.plot(time_orig, bmp_alt, 'o', alpha=0.5, label='BMP Original')
    plt.plot(time_interp, bmp_alt_interp, '-', label='BMP Interpolado')
    plt.xlabel("Tempo [s]")
    plt.ylabel("Altitude [m]")
    plt.title("Interpolação da Altitude BMP")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return bmp_alt_interp

altitude_bmp_interp = interpolar_bmp(t_s, altitude_proc, altitude_filt, time_s_interp, ativar=True)

# ============================================================
# MÓDULO 8 – ZUPT Combinado (Função única com máscara global)
# ============================================================

import math

def aplicar_zupt(t, mpu_accel, mpu_gyro,
                 accel_threshold=0.5, gyro_threshold=2.0,
                 min_stationary_time=0.1,
                 apply_zeroing=True,
                 exibir=True):
    """
    ZUPT detector e aplicador.
    - mpu_accel: Nx3 (com gravidade presente)
    - mpu_gyro: Nx3
    - t: vetor de tempo (s)
    - accel_threshold: tolerância em m/s² para norma da aceleração
    - gyro_threshold: tolerância em °/s para norma do gyro
    - min_stationary_time: duração mínima (s) para considerar bloco estacionário
    - apply_zeroing: se True, zera aceleração e gyro durante períodos detectados
    - exibir: se True, exibe gráficos

    Retorna:
      mpu_accel_zupt, mpu_gyro_zupt, zupt_mask_filtered
    """

    N = mpu_accel.shape[0]
    fs = 1 / np.mean(np.diff(t))  # frequência de amostragem

    # norma dos vetores de aceleração e giroscópio
    accel_norm = np.linalg.norm(mpu_accel, axis=1)
    gyro_norm  = np.linalg.norm(mpu_gyro, axis=1)

    # máscara inicial de estacionariedade
    zupt_mask = ((accel_norm - 9.81) < accel_threshold) & (gyro_norm < gyro_threshold)

    # eliminar pulsos curtos: run-length filtering
    min_len_samples = max(1, int(math.ceil(min_stationary_time * fs)))
    zupt_mask_filtered = np.zeros_like(zupt_mask, dtype=bool)

    start = None
    for i, val in enumerate(zupt_mask):
        if val and start is None:
            start = i
        elif (not val or i == N-1) and start is not None:
            end = i if not val else i+1
            length = end - start
            if length >= min_len_samples:
                zupt_mask_filtered[start:end] = True
            start = None

    # Aplicar zeroing se solicitado
    mpu_accel_zupt = mpu_accel.copy()
    mpu_gyro_zupt  = mpu_gyro.copy()

    if apply_zeroing:
        mpu_accel_zupt[np.abs(mpu_accel_zupt) < accel_threshold] = 0
        mpu_gyro_zupt[np.abs(mpu_gyro_zupt) < gyro_threshold]   = 0
    

    # --- Exibição opcional ---
    if exibir:
        # Gráfico dos sinais originais vs ZUPT
        def plot_comparativo(t, original, processado, nome, unidade):
            fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            eixos = ['X', 'Y', 'Z']
            for i in range(3):
                axs[i].plot(t, original[:, i], 'o-', label='Original', color='gray', alpha=0.5, markersize=2)
                axs[i].plot(t, processado[:, i], '-', label='Com ZUPT', color=plt.cm.viridis(0.2 + i*0.3))
                axs[i].set_ylabel(f'{eixos[i]} ({unidade})')
                axs[i].grid(True, linestyle='--', alpha=0.4)
                axs[i].legend()
            axs[2].set_xlabel('Tempo (s)')
            fig.suptitle(f'{nome} - Original vs ZUPT', fontsize=14, weight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()

        plot_comparativo(t, mpu_accel, mpu_accel_zupt, 'Aceleração MPU6050', 'm/s²')
        plot_comparativo(t, mpu_gyro, mpu_gyro_zupt, 'Velocidade Angular MPU6050', '°/s')

        # Máscara ZUPT
        plt.figure(figsize=(12, 4))
        plt.plot(t, accel_norm, label='|A| (m/s²)', color='tab:blue', alpha=0.8)
        plt.plot(t, gyro_norm / np.max(gyro_norm) * accel_threshold, label='|G| (°/s, normalizado)', color='tab:orange', alpha=0.8)
        plt.fill_between(t, 0, accel_threshold, where=zupt_mask_filtered, color='red', alpha=0.3, label='Repouso detectado (ZUPT)')
        plt.title('ZUPT - Regiões de repouso detectadas')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Magnitude')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Relatório ---
    n_stationary = zupt_mask_filtered.sum()
    pct = 100.0 * n_stationary / N
    print(f"ZUPT aplicado: {n_stationary}/{N} amostras estacionárias ({pct:.2f}%).")
    print(f"Threshold aceleração: {accel_threshold:.3f} m/s²")
    print(f"Threshold giroscópio: {gyro_threshold:.3f} °/s")

    return mpu_accel_zupt, mpu_gyro_zupt, zupt_mask_filtered

mpu_accel_zupt, mpu_gyro_zupt, mask_zupt = aplicar_zupt(
    time_s_interp, mpu_accel_filt, mpu_gyro_filt,
    accel_threshold=0.27, gyro_threshold=2.0,
    min_stationary_time=0.06,
    apply_zeroing=True,
    exibir=True
)

# ===============================================================
# MÓDULO 9 - ORIENTAÇÃO (com média das primeiras amostras)
# ===============================================================
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R


# =============================================================
# 2️⃣ MADGWICK (implementação nativa)
# =============================================================
def madgwick_filter(gyro, accel, time_s, q0, beta=0.5):
    """
    Implementação simplificada do filtro de Madgwick.
    gyro: Nx3 [rad/s]
    accel: Nx3 [m/s²]
    time_s: vetor de tempo [s]
    q0: quaternion inicial [w, x, y, z]
    beta: parâmetro de ganho
    """
    quats = np.zeros((len(time_s), 4))
    quats[0] = q0

    for k in range(1, len(time_s)):
        dt = time_s[k] - time_s[k-1]
        q = quats[k-1]

        # --- Normaliza aceleração ---
        a = accel[k]
        a_norm = np.linalg.norm(a)
        if a_norm == 0:
            quats[k] = q
            continue
        a /= a_norm

        # --- Função objetivo (direção da gravidade) ---
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - a[0],
            2*(q[0]*q[1] + q[2]*q[3]) - a[1],
            2*(0.5 - q[1]**2 - q[2]**2) - a[2]
        ])

        # --- Jacobiano ---
        J = np.array([
            [-2*q[2],  2*q[3], -2*q[0], 2*q[1]],
            [ 2*q[1],  2*q[0],  2*q[3], 2*q[2]],
            [ 0,      -4*q[1], -4*q[2], 0]
        ])

        step = J.T @ f
        step /= np.linalg.norm(step) + 1e-8

        # --- Derivada do quaternion via giroscópio ---
        wx, wy, wz = gyro[k]
        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        dq = 0.5 * Omega @ q

        # --- Correção Madgwick ---
        dq -= beta * step

        # --- Integração e normalização ---
        q = q + dq * dt
        quats[k] = q / np.linalg.norm(q)

    return quats


def obter_orientacao(acc, gyro, time_s, FS, alpha=0.99, ativar_plot=True, n_media_inicial=50):
    """
    Calcula a orientação do sistema (roll, pitch, yaw) a partir dos dados do MPU.
    Utiliza dois métodos: filtro Madgwick e filtro complementar.
    Inclui yaw aproximado via integração do giroscópio e orientação inicial pelo vetor médio da gravidade.

    Parâmetros
    ----------
    acc : ndarray (N x 3)
        Aceleração [g]
    gyro : ndarray (N x 3)
        Velocidade angular [rad/s]
    time_s : ndarray
        Vetor de tempo [s]
    FS : float
        Frequência de amostragem [Hz]
    alpha : float
        Constante do filtro complementar (0.98 padrão)
    ativar_plot : bool
        Exibe ou não os gráficos comparativos
    n_media_inicial : int
        Número de amostras usadas para calcular a orientação inicial

    Retorna
    -------
    quat_madgwick : ndarray (N x 4)
        Quaternions estimados pelo filtro Madgwick
    quat_complementar : ndarray (N x 4)
        Quaternions estimados pelo filtro complementar
    euler_madgwick : ndarray (N x 3)
        Ângulos de Euler [graus] (roll, pitch, yaw) - Madgwick
    euler_complementar : ndarray (N x 3)
        Ângulos de Euler [graus] (roll, pitch, yaw) - Complementar
    """

    # =============================================================
    # 1️⃣ ORIENTAÇÃO INICIAL A PARTIR DA MÉDIA DAS PRIMEIRAS AMOSTRAS
    # =============================================================
    n_inicial = min(n_media_inicial, len(acc))
    acc_media = np.mean(acc[:n_inicial], axis=0)
    ax0, ay0, az0 = acc_media

    roll0 = np.arctan2(ay0, az0)
    pitch0 = np.arctan2(-ax0, np.sqrt(ay0**2 + az0**2))
    yaw0 = 0.0  # sem magnetômetro

    #converter gyro para rad/s se necessário
    gyro = np.deg2rad(gyro)

    # Cria um quaternion inicial com base nesses ângulos
    r_init = R.from_euler('xyz', [roll0, pitch0, yaw0])
    quat_init = r_init.as_quat()  # formato [x, y, z, w]
    quat_init = np.array([quat_init[3], quat_init[0], quat_init[1], quat_init[2]])  # reorganiza p/ [w, x, y, z]

    print(f"🧭 Orientação inicial baseada na média das {n_inicial} primeiras amostras:")
    print(f"Roll0 = {np.degrees(roll0):.2f}°, Pitch0 = {np.degrees(pitch0):.2f}°, Yaw0 = 0°")



    # =============================================================
    # 2️⃣ MADGWICK — APLICAÇÃO AO CONJUNTO DE DADOS
    # =============================================================
    #quat_madgwick = madgwick_filter( gyro=gyro, accel=acc, time_s=time_s, q0=quat_init, beta=0.5 )

    # =============================================================
    # 2️⃣ MADGWICK (com normalização e beta ajustável)
    # =============================================================
    madgwick = Madgwick(sampleperiod=1/FS, beta=0.7)  # beta configurável
    quat_madgwick = np.zeros((len(acc), 4))
    quat_madgwick[0] = quat_init  # orientação inicial

    for t in range(1, len(acc)):
     # Normaliza a aceleração (evita erros por variação de módulo)
     acc_norm = acc[t] / np.linalg.norm(acc[t]) if np.linalg.norm(acc[t]) != 0 else acc[t]

     # Atualiza o filtro Madgwick
     quat_madgwick[t] = madgwick.updateIMU(quat_madgwick[t-1], gyr=gyro[t], acc=acc_norm)


    # Converter para ângulos de Euler (graus)
    r_madgwick = R.from_quat(quat_madgwick[:, [1, 2, 3, 0]])  # [w, x, y, z] → [x, y, z, w]
    euler_madgwick = np.degrees(r_madgwick.as_euler('xyz', degrees=False))


    # =============================================================
    # 3️⃣ FILTRO COMPLEMENTAR
    # =============================================================
    n = len(time_s)
    dt = 1 / FS
    roll_c = np.zeros(n)
    pitch_c = np.zeros(n)
    yaw_c = np.zeros(n)

    # Inicializa com os mesmos ângulos da média da aceleração
    roll_c[0] = np.degrees(roll0)
    pitch_c[0] = np.degrees(pitch0)
    yaw_c[0] = 0.0

    for i in range(1, n):
        # Integração do giroscópio
        roll_gyro = roll_c[i-1] + gyro[i, 0] * dt * 180/np.pi
        pitch_gyro = pitch_c[i-1] + gyro[i, 1] * dt * 180/np.pi
        yaw_gyro = yaw_c[i-1] + gyro[i, 2] * dt * 180/np.pi

        # Ângulos estimados pelo acelerômetro
        roll_acc = np.degrees(np.arctan2(acc[i, 1], acc[i, 2]))
        pitch_acc = np.degrees(np.arctan2(-acc[i, 0], np.sqrt(acc[i, 1]**2 + acc[i, 2]**2)))

        # Filtro complementar
        roll_c[i] = alpha * roll_gyro + (1 - alpha) * roll_acc
        pitch_c[i] = alpha * pitch_gyro + (1 - alpha) * pitch_acc
        yaw_c[i] = yaw_gyro  # integração pura (sem correção magnética)

    euler_complementar = np.vstack((roll_c, pitch_c, yaw_c)).T

    # Converter para quaternions
    r_comp = R.from_euler('xyz', np.radians(euler_complementar))
    quat_complementar = r_comp.as_quat()[:, [3, 0, 1, 2]]  # reorganiza para [w, x, y, z]

    # =============================================================
    # 4️⃣ PLOTAGENS
    # =============================================================
    if ativar_plot:
        # Madgwick
        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
            ax[i].plot(time_s, euler_madgwick[:, i], label=f"{axis} (Madgwick)", color=plt.cm.viridis(0.2 + i * 0.3))
            ax[i].set_ylabel("Graus (°)")
            ax[i].grid(True, linestyle='--', alpha=0.6)
            ax[i].legend()
        ax[-1].set_xlabel("Tempo [s]")
        fig.suptitle("Ângulos de Euler — Filtro Madgwick")
        plt.tight_layout()
        plt.show()

        # Complementar
        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
            ax[i].plot(time_s, euler_complementar[:, i], label=f"{axis} (Complementar)", color=plt.cm.plasma(0.2 + i * 0.3))
            ax[i].set_ylabel("Graus (°)")
            ax[i].grid(True, linestyle='--', alpha=0.6)
            ax[i].legend()
        ax[-1].set_xlabel("Tempo [s]")
        fig.suptitle("Ângulos de Euler — Filtro Complementar")
        plt.tight_layout()
        plt.show()

        # Comparação
        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
            ax[i].plot(time_s, euler_madgwick[:, i], label=f"{axis} Madgwick", color='C0', alpha=0.8)
            ax[i].plot(time_s, euler_complementar[:, i], label=f"{axis} Complementar", color='C1', linestyle='--', alpha=0.8)
            ax[i].set_ylabel("Graus (°)")
            ax[i].grid(True, linestyle='--', alpha=0.6)
            ax[i].legend()
        ax[-1].set_xlabel("Tempo [s]")
        fig.suptitle("Comparação de Orientação — Madgwick × Complementar")
        plt.tight_layout()
        plt.show()

    print("✅ Orientação calculada com sucesso (baseada na média inicial)!")
    return quat_madgwick, quat_complementar, euler_madgwick, euler_complementar

quat_madgwick, quat_complementar, euler_madgwick, euler_complementar = obter_orientacao(
    mpu_accel_zupt, mpu_gyro_zupt, time_s_interp, FS_novo,
    alpha=0.98, ativar_plot=True, n_media_inicial=50
)

# ============================================================
# MÓDULO 9.1 — Correção de Orientação da Aceleração
# ============================================================
from ahrs.common.orientation import q2R

def corrigir_orientacao_aceleracao(acc_mpu, quat_madgwick, quat_complementar=None,
                                   use_madgwick=True, g_ref=9.80665):
    """
    Corrige a orientação da aceleração com base nos quaternions obtidos.

    Parâmetros:
    ------------
    acc_mpu : ndarray (N,3)
        Vetores de aceleração do MPU (no referencial do sensor), em m/s²
    quat_madgwick : ndarray (N,4)
        Quaternions unitários obtidos pelo filtro Madgwick
    quat_complementar : ndarray (N,4), opcional
        Quaternions obtidos pelo filtro complementar
    use_madgwick : bool
        Define se serão usados os quaternions do Madgwick (True) ou do complementar (False)
    g_ref : float
        Aceleração da gravidade, em m/s² (padrão = 9.80665)

    Retorna:
    --------
    acc_global : ndarray (N,3)
        Aceleração rotacionada para o referencial global (sem remoção da gravidade)
    acc_corrigida : ndarray (N,3)
        Aceleração rotacionada com remoção da gravidade
    """

    quat = quat_madgwick if use_madgwick else quat_complementar
    n = len(acc_mpu)
    acc_global = np.zeros_like(acc_mpu)
    acc_corrigida = np.zeros_like(acc_mpu)

    # Gravidade no referencial global
    g_global = np.array([0, 0, g_ref])

    for i in range(n):
        R = q2R(quat[i])        # matriz de rotação do quaternion (sensor → global)
        acc_global[i] = R @ acc_mpu[i]
        acc_corrigida[i] = acc_global[i] - g_global

    # --- Plot comparativo ---
    eixos = ['X', 'Y', 'Z']
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i in range(3):
        axs[i].plot(acc_mpu[:, i], 'o', markersize=2, alpha=0.6, label='Bruta (sensor)')
        axs[i].plot(acc_global[:, i], '-', linewidth=1.2, label='Rotacionada (global)')
        axs[i].plot(acc_corrigida[:, i], '-', color=plt.cm.viridis(0.2 + i*0.3), linewidth=1.8, label='Corrigida (sem gravidade)')
        axs[i].set_ylabel(f"Acel. {eixos[i]} [m/s²]")
        axs[i].grid(True, linestyle='--', alpha=0.6)
        axs[i].legend()
    axs[-1].set_xlabel("Amostra")
    plt.suptitle("Correção de Orientação da Aceleração — MPU6050")
    plt.tight_layout()
    plt.show()

    # --- Plot exibição ---
    eixos = ['X', 'Y', 'Z']
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i in range(3):
        axs[i].plot(acc_corrigida[:, i], '-', color=plt.cm.viridis(0.2 + i*0.3), linewidth=1.8, label='Corrigida (sem gravidade)')
        axs[i].set_ylabel(f"Acel. {eixos[i]} [m/s²]")
        axs[i].grid(True, linestyle='--', alpha=0.6)
        axs[i].legend()
    axs[-1].set_xlabel("Amostra")
    plt.suptitle("Correção de Orientação da Aceleração — MPU6050")
    plt.tight_layout()
    plt.show()

    # correção do ZUTP secundaria (remoção de pequenos valores residuais)
    limiar_zupt = 0.7  # m/s²
    acc_corrigida[np.abs(acc_corrigida) < limiar_zupt] = 0.0

    # --- Plot exibição ---
    eixos = ['X', 'Y', 'Z']
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i in range(3):
        axs[i].plot(acc_corrigida[:, i], '-', color=plt.cm.viridis(0.2 + i*0.3), linewidth=1.8, label='Corrigida (sem gravidade)')
        axs[i].set_ylabel(f"Acel. {eixos[i]} [m/s²]")
        axs[i].grid(True, linestyle='--', alpha=0.6)
        axs[i].legend()
    axs[-1].set_xlabel("Amostra")
    plt.suptitle("Correção de Orientação da Aceleração — MPU6050 + ZUTP Secundária")
    plt.tight_layout()
    plt.show()

    print("\n✅ Correção de orientação concluída.")
    print(f"Modelo utilizado: {'Madgwick' if use_madgwick else 'Complementar'}")
    print("Retornando aceleração rotacionada e aceleração corrigida.\n")

    return acc_global, acc_corrigida

acc_global, acc_corrigida = corrigir_orientacao_aceleracao(
    mpu_accel_zupt, quat_madgwick, quat_complementar,
    use_madgwick=False, g_ref=9.80665
)

# ============================================================
# MÓDULO 10 — Integração, Correção e Deslocamento 3D
# ============================================================

from scipy import integrate, signal

def integrar_movimento(acc_corrigida, tempo_s, zutp_mask, fc_fir=0.5, ordem_fir=801):
    """
    Integra a aceleração corrigida para obter velocidade e deslocamento 3D,
    aplica máscara ZUPT, filtragem FIR passa-alta e gera visualizações.

    Parâmetros:
    ------------
    acc_corrigida : ndarray (N,3)
        Aceleração corrigida (m/s²) no referencial global.
    tempo_s : ndarray (N,)
        Vetor de tempo em segundos (regular).
    zutp_mask : ndarray (N,)
        Máscara booleana indicando onde o sensor está parado (True = parado).
    fc_fir : float
        Frequência de corte do filtro FIR passa-alta (Hz).
    ordem_fir : int
        Ordem do filtro FIR.

    Retorna:
    --------
    vel_filt : ndarray (N,3)
        Velocidade filtrada e corrigida (m/s)
    disp_filt : ndarray (N,3)
        Deslocamento filtrado e corrigido (m)
    """
    print("🔹 Iniciando integração por trapézio...")

    # --- Integração numérica ---
    vel = integrate.cumulative_trapezoid(acc_corrigida, tempo_s, axis=0, initial=0.0)
    disp = integrate.cumulative_trapezoid(vel, tempo_s, axis=0, initial=0.0)

    # --- Aplicar máscara ZUPT à velocidade ---
    vel_masked = vel.copy()
    vel_masked[zutp_mask] = 0.0

    # --- Filtro FIR passa-alta para remover drift ---
    fs = 1 / np.mean(np.diff(tempo_s))
    fir_coeff = signal.firwin(ordem_fir, fc_fir, fs=fs, pass_zero=False)

    vel_filt = np.zeros_like(vel_masked)
    disp_filt = np.zeros_like(disp)

    for i in range(3):
        vel_filt[:, i] = signal.filtfilt(fir_coeff, [1.0], vel_masked[:, i])
        disp_filt[:, i] = integrate.cumulative_trapezoid(vel_filt[:, i], tempo_s, initial=0.0)

    # --- Visualização ---
    eixos = ['X', 'Y', 'Z']
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axs[0].plot(tempo_s, vel[:, i], '--', alpha=0.5, label=f'Velocidade Original {eixos[i]}')
        axs[0].plot(tempo_s, vel_filt[:, i], '-', label=f'Velocidade Filtrada {eixos[i]}')
        axs[1].plot(tempo_s, disp[:, i], '--', alpha=0.5, label=f'Deslocamento Original {eixos[i]}')
        axs[1].plot(tempo_s, disp_filt[:, i], '-', label=f'Deslocamento Filtrado {eixos[i]}')
    axs[0].set_ylabel('Velocidade [m/s]')
    axs[1].set_ylabel('Deslocamento [m]')
    axs[1].set_xlabel('Tempo [s]')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()
    axs[1].legend()
    plt.suptitle('Velocidade e Deslocamento — Pós-ZUPT e Filtragem')
    plt.tight_layout()
    plt.show()

    # --- Visualização 3D da trajetória filtrada ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(disp_filt[:, 0], disp_filt[:, 1], disp_filt[:, 2], 
        linewidth=2, color='purple', label='Trajetória Filtrada')
    ax.scatter(disp_filt[0, 0], disp_filt[0, 1], disp_filt[0, 2], 
           color='green', s=100, label='Início')
    ax.scatter(disp_filt[-1, 0], disp_filt[-1, 1], disp_filt[-1, 2], 
           color='red', s=100, label='Fim')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Trajetória 3D Integrada Pós-ZUPT')
    ax.legend()
    plt.tight_layout()
    plt.show()


    print("✅ Integração concluída e filtragem aplicada com sucesso.")
    return vel_filt, disp_filt

vel_final, disp_final = integrar_movimento(
    acc_corrigida, time_s_interp, mask_zupt,
    fc_fir=0.5, ordem_fir=161
)

# ===================== MÓDULO 12: Compensação ADXL =====================
def compensar_adxl(accel_mpu, accel_adxl, limite_mpu_g=8):
    """
    Substitui valores saturados do MPU pelos valores do ADXL.
    
    Parâmetros:
    ------------
    accel_mpu : ndarray
        Aceleração do MPU (m/s²)
    accel_adxl : ndarray
        Aceleração do ADXL (m/s²)
    limite_mpu_g : float
        Limite da escala do MPU (em g)
    escala_mpu_g : float
        Escala do MPU configurada (em g)
    
    Retorna:
    ---------
    accel_corr : ndarray
        Aceleração corrigida (m/s²)
    """
    print("Compensando furos do MPU com ADXL...")
    limite_ms2 = limite_mpu_g * 9.80665  # converter g -> m/s²
    accel_corr = accel_mpu.copy()
    mask_saturada = np.any(np.abs(accel_mpu) >= limite_ms2, axis=1)
    accel_corr[mask_saturada] = accel_adxl[mask_saturada]
    
    #Plot comparativo
    fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
    eixos = ['X','Y','Z']
    for i in range(3):
        ax[i].plot(accel_mpu[:,i], 'o', markersize=3, alpha=0.5, label='MPU Original')
        ax[i].plot(accel_adxl[:,i], '-', linewidth=1.2, label='ADXL')
        ax[i].plot(accel_corr[:,i], '-', color=plt.cm.viridis(0.2 + i*0.3), linewidth=1.8, label='MPU Compensado')
        ax[i].set_ylabel(f"Acel. {eixos[i]} [m/s²]")
        ax[i].grid(True, linestyle='--', alpha=0.6)
        ax[i].legend()
    ax[-1].set_xlabel("Amostra")
    plt.suptitle("Compensação de Aceleração MPU com ADXL")
    plt.tight_layout()
    plt.show()

    return accel_corr

accel_corr = compensar_adxl(mpu_accel_zupt, adxl_accel_filt, limite_mpu_g=8)



# ===================== MÓDULO 13: Processamento altitude BMP =====================
from scipy.signal import savgol_filter

def processar_altitude_bmp(altitude_bmp, tempo_s, janela=51, polyorder=3):
    """
    Aplica derivada dupla usando Savitzky-Golay para obter aceleração e velocidade vertical.
    
    Parâmetros:
    ------------
    altitude_bmp : ndarray
        Valores de altitude (m)
    tempo_s : ndarray
        Vetor de tempo (s)
    janela : int
        Janela do filtro Savitzky-Golay (ímpar)
    polyorder : int
        Ordem polinomial do filtro
    
    Retorna:
    ---------
    accel_z_bmp : ndarray
        Aceleração vertical estimada (m/s²)
    vel_z_bmp : ndarray
        Velocidade vertical estimada (m/s)
    desloc_z_bmp : ndarray
        Deslocamento vertical (m) - basicamente altitude filtrada
    """
    print("[Módulo 13] Processando altitude do BMP...")
    vel_z_bmp = savgol_filter(altitude_bmp, janela, polyorder, deriv=1, delta=tempo_s[1]-tempo_s[0])
    accel_z_bmp = savgol_filter(altitude_bmp, janela, polyorder, deriv=2, delta=tempo_s[1]-tempo_s[0])
    desloc_z_bmp = altitude_bmp.copy()
    # Plotagem
    fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
    ax[0].plot(tempo_s, accel_z_bmp, label='Aceleração Z BMP', color=plt.cm.viridis(0.3))
    ax[0].set_ylabel("Aceleração [m/s²]")
    ax[1].plot(tempo_s, vel_z_bmp, label='Velocidade Z BMP', color=plt.cm.viridis(0.6))
    ax[1].set_ylabel("Velocidade [m/s]")
    ax[2].plot(tempo_s, desloc_z_bmp, label='Deslocamento Z BMP', color=plt.cm.viridis(0.9))
    ax[2].set_ylabel("Deslocamento [m]")
    ax[2].set_xlabel("Tempo [s]")
    plt.suptitle("Processamento de Altitude - BMP")
    plt.tight_layout()
    plt.show()

    return accel_z_bmp, vel_z_bmp, desloc_z_bmp

accel_z_bmp, vel_z_bmp, desloc_z_bmp = processar_altitude_bmp(
    altitude_bmp_interp, time_s_interp,
    janela=51, polyorder=3
)

# ===================== MÓDULO 14: Reaplicação de orientação =====================

acc_global_adxl, acc_corrigida_adxl = corrigir_orientacao_aceleracao(
    accel_corr, quat_madgwick, quat_complementar,
    use_madgwick=False, g_ref=9.80665
)


# ===================== MÓDULO 15: Integração final com fusão =====================

def integrar_deslocamento_bmp(accel_corr, vel_final, disp_final, accel_z_bmp, vel_z_bmp, desloc_z_bmp, time_s, mask_zutp):
    """
    Integra aceleração para obter velocidade e deslocamento aplicando fusão com BMP,
    filtro FIR, ZUTP e cálculo de deslocamento total e absoluto.
    
    Parâmetros:
    ------------
    accel_corr : ndarray
        Aceleração corrigida 3D (m/s²)
    accel_z_bmp : ndarray
        Aceleração vertical do BMP (m/s²)
    time_s : ndarray
        Tempo (s)
    mask_zutp : ndarray
        Máscara ZUTP
    fc_vel : float
        Frequência de corte do FIR para velocidade (Hz)
    ordem_fir : int
        Ordem do filtro FIR
    
    Retorna:
    ---------
    vel : ndarray
        Velocidade 3D filtrada (m/s)
    desloc : ndarray
        Deslocamento 3D filtrado (m)
    """
    print("[Módulo 15] Calculando velocidade e deslocamento com fusão BMP...")

    #Separar o eixo z da aceleração corrigida, velocidade e deslocamento
    accel_corr_z = accel_corr[:, 2]
    vel_final_z = vel_final[:, 2]
    disp_final_z = disp_final[:, 2]

    #Funcdir com os dados do BMP do eixo z por um filtro comlemntar
    alpha_acel = 1
    alpha_vel = 0.99
    alpha_disp = 0.75

    accel_z_fusao= alpha_acel * accel_corr_z + (1 - alpha_acel) * accel_z_bmp
    vel_z_fusao = alpha_vel * vel_final_z + (1 - alpha_vel) * vel_z_bmp
    disp_z_fusao = alpha_disp * disp_final_z + (1 - alpha_disp) * desloc_z_bmp

    # Juntar os eixos x, y com o z fundido
    accel_fusao = accel_corr.copy()
    accel_fusao[:, 2] = accel_z_fusao

    vel_fusao = vel_final.copy()
    vel_fusao[:, 2] = vel_z_fusao

    disp_fusao = disp_final.copy()
    disp_fusao[:, 2] = disp_z_fusao

    # --- Aplicar máscara ZUPT à velocidade ---
    vel_fusao_masked = vel_fusao.copy()
    vel_fusao_masked[~mask_zutp] = 0
    print("✅ Integração com fusão BMP concluída.")


    # --- Visualização ---
    eixos = ['X', 'Y', 'Z']
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axs[0].plot(time_s, vel_final[:, i], '--', alpha=0.5, label=f'Velocidade Original {eixos[i]}')
        axs[0].plot(time_s, vel_fusao_masked[:, i], '-', label=f'Velocidade com Fusão {eixos[i]}')
        axs[1].plot(time_s, disp_final[:, i], '--', alpha=0.5, label=f'Deslocamento Original {eixos[i]}')
        axs[1].plot(time_s, disp_fusao[:, i], '-', label=f'Deslocamento com Fusão {eixos[i]}')
    axs[0].set_ylabel('Velocidade [m/s]')
    axs[1].set_ylabel('Deslocamento [m]')
    axs[1].set_xlabel('Tempo [s]')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()
    axs[1].legend()
    plt.suptitle('Velocidade e Deslocamento — Pós-Fusão BMP')
    plt.tight_layout()
    plt.show()

    #Definição das variaveis de saída

    accel_final = accel_fusao
    vel_final = vel_fusao_masked
    disp_final = disp_fusao

    return accel_final, vel_final, disp_final

accel_final, vel_final, disp_final = integrar_deslocamento_bmp(
    acc_corrigida_adxl, vel_final, disp_final,
    accel_z_bmp, vel_z_bmp, desloc_z_bmp,
    time_s_interp, mask_zupt
)


# ============================================================
# MÓDULO DE VISUALIZAÇÃO INTERATIVA 3D
# ============================================================

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def visualizar_trajetoria_3d_interativa(tempo_s, accel, vel, disp, exibir_resumo=True, quiver_step=20):
    """
    Visualização e análise completa da trajetória 3D com vetores de velocidade.

    Parâmetros:
    -----------
    tempo_s : ndarray [N]
        Vetor de tempo em segundos.
    accel : ndarray [N x 3]
        Vetor de aceleração (m/s²).
    vel : ndarray [N x 3]
        Vetor de velocidade (m/s).
    disp : ndarray [N x 3]
        Vetor de deslocamento (m).
    exibir_resumo : bool
        Se True, exibe informações resumidas da trajetória.
    quiver_step : int
        Passo para plotar vetores de velocidade (reduz a densidade).

    Retorna:
    --------
    None (gera gráficos e prints das métricas)
    """

    # --- Estatísticas ---
    deslocamento_abs = disp[-1] - disp[0]
    distancia_total = np.sum(np.linalg.norm(np.diff(disp, axis=0), axis=1))
    
    vel_mod = np.linalg.norm(vel, axis=1)
    vel_max = np.max(vel_mod)
    vel_med = np.mean(vel_mod)
    
    accel_mod = np.linalg.norm(accel, axis=1)
    accel_max = np.max(accel_mod)
    accel_med = np.mean(accel_mod)
    
    tempo_total = tempo_s[-1] - tempo_s[0]

    # --- Resumo ---
    if exibir_resumo:
        print("===== RESUMO DA TRAJETÓRIA =====")
        print(f"Deslocamento absoluto (vetor): {deslocamento_abs}")
        print(f"Distância total percorrida (trajetória em arco): {distancia_total:.3f} m")
        print(f"Velocidade máxima: {vel_max:.3f} m/s")
        print(f"Velocidade média: {vel_med:.3f} m/s")
        print(f"Aceleração máxima: {accel_max:.3f} m/s²")
        print(f"Aceleração média: {accel_med:.3f} m/s²")
        print(f"Tempo total: {tempo_total:.3f} s")
        print("===============================")

    # --- Gráfico 2D por eixo ---
    eixos = ['X', 'Y', 'Z']
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for i in range(3):
        axs[0].plot(tempo_s, vel[:, i], '-', label=f'Vel {eixos[i]}')
        axs[1].plot(tempo_s, disp[:, i], '-', label=f'Disp {eixos[i]}')
        axs[2].plot(tempo_s, accel[:, i], '-', label=f'Accel {eixos[i]}')
    
    axs[0].set_ylabel('Velocidade [m/s]')
    axs[1].set_ylabel('Deslocamento [m]')
    axs[2].set_ylabel('Aceleração [m/s²]')
    axs[2].set_xlabel('Tempo [s]')
    
    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
    
    plt.suptitle('Trajetória 3D — Análise de Eixos', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()

    # --- Gráfico 3D interativo ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Trajetória colorida por velocidade
    vel_norm = (vel_mod - np.min(vel_mod)) / (np.max(vel_mod) - np.min(vel_mod) + 1e-6)
    for i in range(len(disp)-1):
        ax.plot(disp[i:i+2,0], disp[i:i+2,1], disp[i:i+2,2], 
                color=cm.viridis(vel_norm[i]), linewidth=2)
    
    # Vetores de velocidade (quiver) para análise visual
    ax.quiver(disp[::quiver_step,0], disp[::quiver_step,1], disp[::quiver_step,2],
              vel[::quiver_step,0], vel[::quiver_step,1], vel[::quiver_step,2],
              length=0.1*distancia_total, color='black', normalize=True, alpha=0.6)
    
    # Pontos de início e fim
    ax.scatter(disp[0,0], disp[0,1], disp[0,2], color='green', s=100, label='Início')
    ax.scatter(disp[-1,0], disp[-1,1], disp[-1,2], color='red', s=100, label='Fim')
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Trajetória 3D Interativa — Velocidade em cores e vetores')
    
    # Barra de cores
    mappable = cm.ScalarMappable(cmap='viridis')
    mappable.set_array(vel_mod)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Velocidade [m/s]')
    
    ax.legend()
    plt.tight_layout()
    plt.show()

visualizar_trajetoria_3d_interativa(
    time_s_interp, accel_final, vel_final, disp_final,
    exibir_resumo=True, quiver_step=5
)



