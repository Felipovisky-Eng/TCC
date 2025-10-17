# Código para receber dados do IMU 6050 com DMP via serial e salvar em CSV, além de plotar gráficos de tempo e jitter.

#Código para analisar a coleta do IMU 6050 com dmp pela serial 

import serial
import struct
import csv
import time
import matplotlib.pyplot as plt
from tqdm import tqdm  # barra de progresso

# Configurações
PORTA = 'COM5'  # Ajuste conforme necessário
BAUDRATE = 921600 # Velocidade da serial que melhor funcionou
MAX_SAMPLES = 12000 # Número máximo de amostras a coletar, esse valor vai limitar a coleta (Lembre-se de ajustar no código do ESP32 também)


# Formato da struct do ESP32
PACKET_FORMAT = '<I4f3h3h3h3h3f3f'  # timestamp, quat[4], accel[3], gyro[3], accel_real[3], accel_world[3], gravity[3], ypr[3]
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

def main():
    # Conecta na serial
    ser = serial.Serial(PORTA, BAUDRATE, timeout=1)
    print("Conectando e aguardando READY do ESP32...")
    
    # Aguarda sinal READY
    while True:
        line = ser.readline().strip()
        if line == b'READY':
            print("ESP32 pronto! Iniciando coleta...")
            break

    dados = []
    timestamps = []
    start_time = time.time()

    # Recebe pacotes com barra de progresso
    for _ in tqdm(range(MAX_SAMPLES), desc="Coletando dados"):
        pkt = ser.read(PACKET_SIZE)
        if len(pkt) != PACKET_SIZE:
            continue
        unpacked = struct.unpack(PACKET_FORMAT, pkt)
        dados.append(unpacked)
        timestamps.append(unpacked[0])

    ser.close()
    end_time = time.time()
    duracao = end_time - start_time
    freq_media = len(dados) / duracao

    print(f"\nColeta concluída em {duracao:.2f} s, frequência média: {freq_media:.1f} Hz")

    # Salva CSV
    with open('dados_imu.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp_us',
            'quat_w','quat_x','quat_y','quat_z',
            'accel_x','accel_y','accel_z',
            'gyro_x','gyro_y','gyro_z',
            'accel_real_x','accel_real_y','accel_real_z',
            'accel_world_x','accel_world_y','accel_world_z',
            'gravity_x','gravity_y','gravity_z',
            'yaw','pitch','roll'
        ])
        writer.writerows(dados)

    # O nome do arquivo CSV foi alterado para dados_imu.csv, ele sobrepoe o arquivo anterior a cada execução, então mude o nome se quiser salvar várias coletas
    print("Arquivo dados_imu.csv salvo com sucesso!")

    # Gráficos de tempo e jitter
    if len(timestamps) > 1:
        ts_s = [(t - timestamps[0])/1e6 for t in timestamps]
        dt_ms = [(ts_s[i+1] - ts_s[i])*1000 for i in range(len(ts_s)-1)]

        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.plot(ts_s, marker='.', markersize=3)
        plt.title("Tempo das Amostras")
        plt.ylabel("Tempo (s)")
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(dt_ms, marker='.', markersize=3)
        plt.title("Jitter entre Amostras")
        plt.xlabel("Amostra")
        plt.ylabel("Intervalo (ms)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
