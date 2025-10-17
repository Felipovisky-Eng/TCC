// Código simples para envio dos dados da DMP para o computador pela serial
#include <Wire.h>
#include <MPU6050_6Axis_MotionApps20.h>

// ========================================================
// CONFIGURAÇÕES AJUSTÁVEIS
// ========================================================
#define SERIAL_BAUDRATE 921600
#define ACCEL_RANGE MPU6050_ACCEL_FS_8
#define GYRO_RANGE MPU6050_GYRO_FS_1000
#define DLPF_BANDWIDTH MPU6050_DLPF_BW_188   // filtro compatível com 200Hz
#define SAMPLE_RATE 200
#define INTERRUPT_PIN 4

MPU6050 mpu;

// ========================================================
// ESTRUTURA DE DADOS
// ========================================================
#pragma pack(push, 1)
struct SensorData {
    uint32_t timestamp;        // µs
    struct { float w,x,y,z; } quat;
    struct { int16_t x,y,z; } accel;
    struct { int16_t x,y,z; } gyro;
    struct { int16_t x,y,z; } accel_real;
    struct { int16_t x,y,z; } accel_world;
    struct { float x,y,z; } gravity;
    struct { float yaw,pitch,roll; } ypr;
};
#pragma pack(pop)

// ========================================================
// VARIÁVEIS GLOBAIS
// ========================================================
volatile bool mpuInterrupt = false;
bool dmpReady = false;
uint16_t packetSize;
uint16_t fifoCount;
uint8_t fifoBuffer[64];

// Estruturas para dados
Quaternion q;
VectorInt16 aa, aaReal, aaWorld, gy;
VectorFloat gravity;
float ypr[3];

// ========================================================
// INTERRUPÇÃO
// ========================================================
void IRAM_ATTR dmpDataReady() {
  mpuInterrupt = true;
}

// ========================================================
// SETUP
// ========================================================
void setup() {
    Serial.begin(SERIAL_BAUDRATE);
    Wire.begin();
    Wire.setClock(400000);

    pinMode(INTERRUPT_PIN, INPUT);
    mpu.initialize();


    mpu.setFullScaleAccelRange(ACCEL_RANGE);
    mpu.setFullScaleGyroRange(GYRO_RANGE);
    mpu.setDLPFMode(DLPF_BANDWIDTH);
    mpu.setRate((1000 / SAMPLE_RATE) - 1);

    // Calibração (ajuste para seu sensor)
    //mpu.setXGyroOffset(0);
    //mpu.setYGyroOffset(0);
    //mpu.setZGyroOffset(0);
    //mpu.setXAccelOffset(0);
    //mpu.setYAccelOffset(0);
    //mpu.setZAccelOffset(0);

    if (mpu.dmpInitialize() == 0) {
        mpu.setDMPEnabled(true);
        attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
        packetSize = mpu.dmpGetFIFOPacketSize();
        dmpReady = true;
        Serial.println("DMP OK");
    } else {
        Serial.println("Falha DMP");
    }

    mpu.setFullScaleAccelRange(ACCEL_RANGE);
    mpu.setFullScaleGyroRange(GYRO_RANGE);
    mpu.setDLPFMode(DLPF_BANDWIDTH);
    mpu.setRate((1000 / SAMPLE_RATE) - 1);






    Serial.println("READY"); // sinal para o Python iniciar

}

// ========================================================
// LOOP
// ========================================================
void loop() {
    if (!dmpReady) return;

    fifoCount = mpu.getFIFOCount();

    // Espera interrupção ou pacote completo
    if (!mpuInterrupt && fifoCount < packetSize) return;
    mpuInterrupt = false;

    // Verifica overflow
    uint8_t mpuIntStatus = mpu.getIntStatus();
    if ((mpuIntStatus & 0x10) || fifoCount >= 1024) {
        mpu.resetFIFO();
        return;
    }

    // Processa pacote
    while (fifoCount >= packetSize) {
        uint32_t timestamp = micros(); // timestamp preciso
        mpu.getFIFOBytes(fifoBuffer, packetSize);
        fifoCount -= packetSize;

        mpu.dmpGetQuaternion(&q, fifoBuffer);
        mpu.dmpGetAccel(&aa, fifoBuffer);
        mpu.dmpGetGyro(&gy, fifoBuffer);
        mpu.dmpGetGravity(&gravity, &q);
        mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
        mpu.dmpGetLinearAccelInWorld(&aaWorld, &aaReal, &q);
        mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);

        // Preenche struct
        SensorData data;
        data.timestamp = timestamp;
        data.quat = {q.w, q.x, q.y, q.z};
        data.accel = {aa.x, aa.y, aa.z};
        data.gyro = {gy.x, gy.y, gy.z};
        data.accel_real = {aaReal.x, aaReal.y, aaReal.z};
        data.accel_world = {aaWorld.x, aaWorld.y, aaWorld.z};
        data.gravity = {gravity.x, gravity.y, gravity.z};
        data.ypr = {ypr[0], ypr[1], ypr[2]};

        // Envia pacote binário
        Serial.write(reinterpret_cast<uint8_t*>(&data), sizeof(data));
    }
}
