import csv
import matplotlib.pyplot as plt

# ================= CARREGA CSV =================
timestamps = []
mpu_accel = []
mpu_gyro = []
adxl_accel = []
altitude = []

with open('dados_sensores.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ts = int(row['timestamp_us'])
        timestamps.append(ts)
        mpu_accel.append([int(row['mpu_ax']), int(row['mpu_ay']), int(row['mpu_az'])])
        mpu_gyro.append([int(row['mpu_gx']), int(row['mpu_gy']), int(row['mpu_gz'])])
        adxl_accel.append([int(row['adxl_x']), int(row['adxl_y']), int(row['adxl_z'])])
        altitude.append(float(row['altitude']))

# ================= CALCULA JITTER =================
timestamps_s = [(t - timestamps[0])/1e6 for t in timestamps]  # segundos
dt_ms = [(timestamps_s[i+1] - timestamps_s[i])*1000 for i in range(len(timestamps_s)-1)]  # ms

# ================= PLOT =================
plt.figure(figsize=(12, 8))

# Tempo das amostras
plt.subplot(3,1,1)
plt.plot(timestamps_s, marker='.', markersize=3)
plt.title("Tempo das Amostras")
plt.ylabel("Tempo (s)")
plt.grid(True)

# Jitter entre amostras
plt.subplot(3,1,2)
plt.plot(dt_ms, marker='.', markersize=3)
plt.title("Jitter entre Amostras")
plt.ylabel("Intervalo (ms)")
plt.grid(True)

# Valores dos sensores
plt.subplot(3,1,3)
plt.plot([v[0] for v in mpu_accel], label='MPU Ax')
plt.plot([v[1] for v in mpu_accel], label='MPU Ay')
plt.plot([v[2] for v in mpu_accel], label='MPU Az')
plt.plot([v[0] for v in adxl_accel], label='ADXL X', linestyle='--')
plt.plot([v[1] for v in adxl_accel], label='ADXL Y', linestyle='--')
plt.plot([v[2] for v in adxl_accel], label='ADXL Z', linestyle='--')
plt.plot(altitude, label='Altitude', linestyle=':')
plt.title("Sensores")
plt.xlabel("Amostra")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
