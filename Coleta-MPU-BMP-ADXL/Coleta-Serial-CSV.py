# Código para recer dados pela serial e plotar graficos de jitter tempo e dados brutos do código VAISSE.ino

import serial
import struct
import csv
import time
from tqdm import tqdm

# ================= CONFIGURAÇÕES =================
PORTA = 'COM5'
BAUDRATE = 921600
MAX_SAMPLES = 2000   # Número máximo de amostras (ou tempo de gravação aproximado)

PACKET_FORMAT = '<I3h3h3hf'  # timestamp, mpu_accel[3], mpu_gyro[3], adxl_accel[3], altitude
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

def main():
    # Conecta à serial
    ser = serial.Serial(PORTA, BAUDRATE, timeout=1)
    print("Aguardando READY do ESP32...")

    # Espera sinal READY
    while True:
        line = ser.readline().strip()
        if line == b'READY':
            print("ESP32 pronto! Iniciando coleta...")
            break

    dados = []
    timestamps = []
    start_time = time.time()

    # Recebe pacotes
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
    with open('dados_sensores.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp_us',
            'mpu_ax','mpu_ay','mpu_az',
            'mpu_gx','mpu_gy','mpu_gz',
            'adxl_x','adxl_y','adxl_z',
            'altitude'
        ])
        writer.writerows(dados)

    print("Arquivo dados_sensores.csv salvo com sucesso!")

if __name__ == "__main__":
    main()
