// Código para visualização e cofiguração simples da DMP com o ESP32


#include <Wire.h>
#include <MPU6050_6Axis_MotionApps20.h>  // Biblioteca com suporte à DMP

MPU6050 mpu;

// ========================================================
// CONFIGURAÇÕES AJUSTÁVEIS
// ========================================================
#define ACCEL_RANGE MPU6050_ACCEL_FS_8      // Escala do acelerômetro (8G)
#define GYRO_RANGE MPU6050_GYRO_FS_1000     // Escala do giroscópio (1000°/s)
#define DLPF_BANDWIDTH MPU6050_DLPF_BW_188  // Largura de banda do filtro (256Hz)
#define SAMPLE_RATE 201                     // Taxa de amostragem (200Hz)

/* Opções de escala disponíveis:
   - Acelerômetro:
     MPU6050_ACCEL_FS_2  (±2G)
     MPU6050_ACCEL_FS_4  (±4G)
     MPU6050_ACCEL_FS_8  (±8G)
     MPU6050_ACCEL_FS_16 (±16G)

   - Giroscópio:
     MPU6050_GYRO_FS_250  (±250°/s)
     MPU6050_GYRO_FS_500  (±500°/s)
     MPU6050_GYRO_FS_1000 (±1000°/s)
     MPU6050_GYRO_FS_2000 (±2000°/s)

   - Opções de DLPF (Digital Low Pass Filter):
     MPU6050_DLPF_BW_256 (256Hz) - Delay: 0.98ms
     MPU6050_DLPF_BW_188 (188Hz) - Delay: 1.9ms
     MPU6050_DLPF_BW_98  (98Hz)  - Delay: 2.8ms
     MPU6050_DLPF_BW_42  (42Hz)  - Delay: 4.8ms
     MPU6050_DLPF_BW_20  (20Hz)  - Delay: 8.3ms
     MPU6050_DLPF_BW_10  (10Hz)  - Delay: 13.4ms
*/

// ========================================================
// VARIÁVEIS GLOBAIS
// ========================================================
bool dmpReady = false;
uint8_t mpuIntStatus;
uint8_t devStatus;
uint16_t packetSize;
uint16_t fifoCount;
uint8_t fifoBuffer[64];

// Estruturas para dados da DMP
Quaternion q;           // [w, x, y, z]
VectorInt16 aa;         // Aceleração bruta
VectorInt16 aaReal;     // Aceleração sem gravidade
VectorInt16 aaWorld;    // Aceleração no referencial do mundo (saída desejada)
VectorFloat gravity;    // Vetor gravidade [x, y, z]
VectorInt16 gy;

float ypr[3];
// Pino de interrupção (conectar o pino INT do MPU6050)
const int INTERRUPT_PIN = 4;
volatile bool mpuInterrupt = false;

// ========================================================
// FUNÇÕES
// ========================================================
void IRAM_ATTR dmpDataReady() {
  mpuInterrupt = true;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);  // I2C em 400kHz

  mpu.initialize();
  pinMode(INTERRUPT_PIN, INPUT);

  // Verifica conexão
  Serial.println(mpu.testConnection() ? "MPU6050 conectado!" : "Falha na conexão");

  // Configura ranges e filtros
  mpu.setFullScaleAccelRange(ACCEL_RANGE);
  mpu.setFullScaleGyroRange(GYRO_RANGE);
  mpu.setDLPFMode(DLPF_BANDWIDTH);
  mpu.setRate(1000 / SAMPLE_RATE - 1);  // Configura taxa de amostragem

  // Inicializa a DMP
  devStatus = mpu.dmpInitialize();

  // Offsets de calibração (ajuste para seu sensor!)
  //mpu.setXGyroOffset(0);
 // mpu.setYGyroOffset(0);
  //mpu.setZGyroOffset(0);
  //mpu.setXAccelOffset(0);
  //mpu.setYAccelOffset(0);
  //mpu.setZAccelOffset(0);

  if (devStatus == 0) {
    mpu.setDMPEnabled(true);
    attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
    mpuIntStatus = mpu.getIntStatus();
    dmpReady = true;
    packetSize = mpu.dmpGetFIFOPacketSize();
    Serial.println("DMP configurada com sucesso!");
  } else {
    Serial.print("Erro na inicialização da DMP (Código ");
    Serial.print(devStatus);
    Serial.println(")");
  }



  mpu.setFullScaleAccelRange(ACCEL_RANGE);
  mpu.setFullScaleGyroRange(GYRO_RANGE);
  mpu.setDLPFMode(DLPF_BANDWIDTH);
  mpu.setRate(1000 / SAMPLE_RATE - 1);  // Configura taxa de amostragem
  

}

void loop() {
  if (!dmpReady) return;

  // Espera por interrupção ou dados disponíveis
  while (!mpuInterrupt && fifoCount < packetSize) {}
  mpuInterrupt = false;

  // Verifica status
  mpuIntStatus = mpu.getIntStatus();
  fifoCount = mpu.getFIFOCount();

  // Trata overflow
  if ((mpuIntStatus & 0x10) || fifoCount == 1024) {
    mpu.resetFIFO();
    Serial.println("FIFO overflow!");
    return;
  }

  // Processa dados quando disponíveis
  if (mpuIntStatus & 0x02) {
    while (fifoCount < packetSize) fifoCount = mpu.getFIFOCount();
    mpu.getFIFOBytes(fifoBuffer, packetSize);
    fifoCount -= packetSize;

    // Obtém os dados da DMP
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetAccel(&aa, fifoBuffer);
    mpu.dmpGetGyro(&gy, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
    mpu.dmpGetLinearAccelInWorld(&aaWorld, &aaReal, &q);
    mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);

   // CONVERTE YPR PARA GRAUS
    float yaw_deg = ypr[0] * 180.0 / M_PI;
    float pitch_deg = ypr[1] * 180.0 / M_PI;
    float roll_deg = ypr[2] * 180.0 / M_PI;

    Serial.print("AccX:"); Serial.print(aa.x);
    Serial.print(",AccY:"); Serial.print(aa.y);
    Serial.print(",AccZ:"); Serial.print(aa.z);
    
    //Serial.print(",GyroX:"); Serial.print(gy.x);
    //Serial.print(",GyroY:"); Serial.print(gy.y);
    //Serial.print(",GyroZ:"); Serial.print(gy.z);

    // Formato para Serial Plotter: label:valor,label:valor,...
    Serial.print(",WorldX:"); Serial.print(aaWorld.x);
    Serial.print(",WorldY:"); Serial.print(aaWorld.y);
    Serial.print(",WorldZ:"); Serial.print(aaWorld.z);

    Serial.print(",Yaw:"); Serial.print(yaw_deg);
    Serial.print(",Pitch:"); Serial.print(pitch_deg);
    Serial.print(",Roll:"); Serial.println(roll_deg);
    

  }
}