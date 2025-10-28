#Código para converter arquivo binário de dados do código VAISSE.ino para CSV

import struct
import csv
import numpy as np
from tkinter import Tk, filedialog
import os

# --- 1. Selecionar arquivo binário ---
Tk().withdraw()
file_path = filedialog.askopenfilename(
    title="Selecione o arquivo .bin",
    filetypes=[("Bin files", "*.bin")]
)
if not file_path:
    print("Nenhum arquivo selecionado.")
    exit()

# --- 2. Ler dados binários ---
# Estrutura do SensorData: timestamp (uint32), mpu_accel[3], mpu_gyro[3], adxl_accel[3], altitude (float)
packet_format = '<I3h3h3hf'
packet_size = struct.calcsize(packet_format)

data = []

with open(file_path, 'rb') as f:
    while True:
        packet = f.read(packet_size)
        if len(packet) < packet_size:
            break
        unpacked = struct.unpack(packet_format, packet)
        data.append(unpacked)

data = np.array(data)
print(f"{len(data)} amostras carregadas.")

# --- 3. Separar colunas ---
timestamps_us = data[:, 0].astype(np.uint32)
mpu_accel = data[:, 1:4].astype(np.int16)
mpu_gyro = data[:, 4:7].astype(np.int16)
adxl_accel = data[:, 7:10].astype(np.int16)
altitude = data[:, 10].astype(np.float32)

# --- 4. Criar arquivo CSV com mesmo nome ---
csv_path = os.path.splitext(file_path)[0] + '.csv'

with open(csv_path, 'w', newline='') as f_csv:
    writer = csv.writer(f_csv)
    # Cabeçalho
    writer.writerow([
        'timestamp_us',
        'mpu_ax','mpu_ay','mpu_az',
        'mpu_gx','mpu_gy','mpu_gz',
        'adxl_x','adxl_y','adxl_z',
        'altitude'
    ])
    # Dados
    for i in range(len(data)):
        writer.writerow([
            timestamps_us[i],
            mpu_accel[i,0], mpu_accel[i,1], mpu_accel[i,2],
            mpu_gyro[i,0], mpu_gyro[i,1], mpu_gyro[i,2],
            adxl_accel[i,0], adxl_accel[i,1], adxl_accel[i,2],
            altitude[i]
        ])

print(f"Arquivo CSV criado: {csv_path}")
