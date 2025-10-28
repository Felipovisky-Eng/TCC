/*********************************************************************
 * Projeto ESP32: Gravação de dados de sensores (MPU6050, ADXL375, BMP580)
 * Funcionalidades:
 * - Gravação no cartão SD com buffer configurável
 * - Transmissão binária via Serial
 * - Exibição serial simples a 5 Hz
 * - LEDs indicam estado do sistema
 * - Botões:
 *     BTN1 - Gravação SD
 *     BTN2 - Transmissão Serial binária
 *     BTN3 - Monitor Serial 5 Hz
 *********************************************************************/

#include <Arduino.h>
#include <Wire.h>
#include <SD.h>
#include <MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL375.h>
#include <Adafruit_BMP5xx.h>

// ======================== PINOS ========================
#define SDA_PIN 21
#define SCL_PIN 22

#define LED_D25 25
#define LED_D32 32
#define LED_D33 33

#define BTN1 13
#define BTN2 14
#define BTN3 26

#define SD_CS 5

// ======================== CONFIGURAÇÕES ========================
#define SAMPLE_HZ 200
#define SAMPLE_PERIOD_US (1000000 / SAMPLE_HZ)
#define DEBOUNCE_MS 50
#define RECORD_DURATION_MS 30000
#define MONITOR_DURATION_MS 10000
#define LONG_PRESS_MS 2000
#define SEALEVELPRESSURE_HPA 1013.25
#define DLPF_BANDWIDTH MPU6050_DLPF_BW_188   // filtro compatível com 200Hz

#define BUFFER_SIZE 20  // Configurável: número de amostras antes de gravar

// ======================== STRUCT DE DADOS ========================
#pragma pack(push, 1)
struct SensorData {
    uint32_t timestamp;     
    int16_t mpu_accel[3];   
    int16_t mpu_gyro[3];    
    int16_t adxl_accel[3];  
    float altitude;         
};
#pragma pack(pop)

// ======================== VARIÁVEIS GLOBAIS ========================
MPU6050 mpu(0x69, &Wire);
Adafruit_ADXL375 adxl(12345);
Adafruit_BMP5xx bmp;

File dataFile;
String fileName;
int nextFileNumber = 1;

bool fileCreated = false;
unsigned long ledFlashTime = 0;
bool ledState = false;

SensorData buffer[BUFFER_SIZE];
int bufferIndex = 0;

// ======================== ESTADOS ========================
enum SystemState {STATE_IDLE, STATE_RECORD_SD, STATE_SERIAL_BIN, STATE_MONITOR_5HZ, STATE_ERROR};
volatile SystemState currentState = STATE_IDLE;

// ======================== BOTÕES ========================
struct Button {
    uint8_t pin;
    bool lastState;
    bool pressedFlag;
    unsigned long lastDebounceTime;
};
Button btn1 = {BTN1, HIGH, false, 0};
Button btn2 = {BTN2, HIGH, false, 0};
Button btn3 = {BTN3, HIGH, false, 0};

// ======================== PROTÓTIPOS ========================
void initSensors();
void readButtons();
void updateLEDs();
bool abrirNovoArquivoSD();
void recordToSD();
void sendSerialBinary();
void serialMonitor();
void printSystemStatus();

// ======================== SETUP ========================
void setup() {
    Serial.begin(921600);
    while(!Serial);

    pinMode(LED_D25, OUTPUT);
    pinMode(LED_D32, OUTPUT);
    pinMode(LED_D33, OUTPUT);

    pinMode(2, OUTPUT);
    pinMode(4, OUTPUT);
    digitalWrite(2, HIGH);
    digitalWrite(4, HIGH);


    pinMode(BTN1, INPUT_PULLUP);
    pinMode(BTN2, INPUT_PULLUP);
    pinMode(BTN3, INPUT_PULLUP);

    Wire.begin(SDA_PIN, SCL_PIN, 400000);
    delay(200);

    initSensors();

    digitalWrite(LED_D25, HIGH);
    digitalWrite(LED_D32, LOW);
    digitalWrite(LED_D33, LOW);

    Serial.println("=== Sistema pronto ===");
}

// ======================== LOOP ========================
void loop() {
    readButtons();
    updateLEDs();

    switch (currentState) {
        case STATE_IDLE:
            static unsigned long lastStatus = 0;
            if (millis() - lastStatus >= 2000) {
                lastStatus = millis();
                printSystemStatus();
            }
            break;

        case STATE_RECORD_SD:
            recordToSD();
            break;

        case STATE_SERIAL_BIN:
            sendSerialBinary();
            break;

        case STATE_MONITOR_5HZ:
            serialMonitor();
            break;

        case STATE_ERROR:
            if (millis() - ledFlashTime > 500) {
                ledFlashTime = millis();
                ledState = !ledState;
                digitalWrite(LED_D25, ledState);
                digitalWrite(LED_D32, ledState);
                digitalWrite(LED_D33, ledState);
            }
            break;
    }

    vTaskDelay(10 / portTICK_PERIOD_MS);
}

// ======================== FUNÇÕES ========================
void initSensors() {
    Serial.println("Inicializando BMP580...");
    if (!bmp.begin(BMP5XX_ALTERNATIVE_ADDRESS, &Wire)) {
        Serial.println("Erro BMP580!");
        currentState = STATE_ERROR;
        return;
    }
    bmp.setTemperatureOversampling(BMP5XX_OVERSAMPLING_2X);
    bmp.setPressureOversampling(BMP5XX_OVERSAMPLING_4X);
    bmp.setIIRFilterCoeff(BMP5XX_IIR_FILTER_COEFF_1);
    bmp.setOutputDataRate(BMP5XX_ODR_218_5_HZ);
    bmp.setPowerMode(BMP5XX_POWERMODE_CONTINUOUS);
    delay(200);

    Serial.println("Inicializando MPU6050...");
    mpu.initialize();
    if (!mpu.testConnection()) {
        Serial.println("Erro MPU6050!");
        currentState = STATE_ERROR;
        return;
    }
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_8);
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_1000);
    mpu.setDLPFMode(DLPF_BANDWIDTH);
    mpu.setRate(4);
    delay(200);

    Serial.println("Inicializando ADXL375...");
    if (!adxl.begin()) {
        Serial.println("Erro ADXL375!");
        currentState = STATE_ERROR;
        return;
    }
    adxl.setTrimOffsets(1,2,-2);
    delay(200);
}

// ======================== BOTÕES ========================
void readButtons() {
    Button *buttons[] = {&btn1, &btn2, &btn3};
    for(int i=0;i<3;i++){
        bool reading = digitalRead(buttons[i]->pin);
        if(reading != buttons[i]->lastState){
            buttons[i]->lastDebounceTime = millis();
        }
        if((millis() - buttons[i]->lastDebounceTime) > DEBOUNCE_MS){
            if(reading != buttons[i]->pressedFlag){
                buttons[i]->pressedFlag = reading;
                if(reading == LOW){
                    switch(i){
                        case 0: currentState = STATE_RECORD_SD; fileCreated=false; break;
                        case 1: currentState = STATE_SERIAL_BIN; break;
                        case 2: currentState = STATE_MONITOR_5HZ; break;
                    }
                }
            }
        }
        buttons[i]->lastState = reading;
    }
}

// ======================== LED ========================
void updateLEDs() {
    switch(currentState){
        case STATE_IDLE: digitalWrite(LED_D25,HIGH); digitalWrite(LED_D32,LOW); digitalWrite(LED_D33,LOW); break;
        case STATE_RECORD_SD: digitalWrite(LED_D32,HIGH); digitalWrite(LED_D33,HIGH); break;
        case STATE_SERIAL_BIN: digitalWrite(LED_D32,HIGH); break;
        case STATE_MONITOR_5HZ: digitalWrite(LED_D33,HIGH); break;
        case STATE_ERROR: /* piscando no loop */ break;
    }
}

// ======================== SD ========================
bool abrirNovoArquivoSD() {
    Serial.println("\n[SD] Inicializando cartão SD...");
    if (!SD.begin(SD_CS)) {
        Serial.println("[SD] Erro ao inicializar o cartão!");
        return false;
    }

    // Procura próximo nome disponível
    int fileCount = 1;
    char fileNameChar[20];
    while (fileCount < 1000) {
        sprintf(fileNameChar, "/DATA%03d.BIN", fileCount);
        if (!SD.exists(fileNameChar)) break;
        fileCount++;
    }

    if (fileCount >= 1000) {
        Serial.println("[SD] Limite máximo de arquivos atingido (999)");
        return false;
    }

    fileName = String(fileNameChar);
    dataFile = SD.open(fileNameChar, FILE_WRITE);
    if (!dataFile) {
        Serial.print("[SD] Erro ao criar o arquivo ");
        Serial.println(fileName);
        return false;
    }

    nextFileNumber = fileCount + 1;
    Serial.println("[SD] Cartão inicializado com sucesso!");
    Serial.print("[SD] Novo arquivo criado: ");
    Serial.println(fileName);
    Serial.println("-----------------------------------");

    return true;
}

// ======================== GRAVAÇÃO SD ========================
void recordToSD() {
    if (!fileCreated) {
        if (!abrirNovoArquivoSD()) {
            currentState = STATE_ERROR;
            return;
        }
        fileCreated = true;
        bufferIndex = 0;
    }

    unsigned long startMillis = millis();
    unsigned long lastSampleMicros = micros();
    ledFlashTime = millis();

    while (currentState == STATE_RECORD_SD) {
        unsigned long nowMicros = micros();
        if (nowMicros - lastSampleMicros >= SAMPLE_PERIOD_US) {
            lastSampleMicros += SAMPLE_PERIOD_US;

            SensorData data;
            data.timestamp = micros();

            int16_t ax, ay, az, gx, gy, gz;
            mpu.getMotion6(&ax,&ay,&az,&gx,&gy,&gz);
            data.mpu_accel[0] = ax; data.mpu_accel[1] = ay; data.mpu_accel[2] = az;
            data.mpu_gyro[0]  = gx; data.mpu_gyro[1]  = gy; data.mpu_gyro[2]  = gz;

            data.adxl_accel[0] = adxl.getX();
            data.adxl_accel[1] = adxl.getY();
            data.adxl_accel[2] = adxl.getZ();

            if (bmp.performReading()) data.altitude = bmp.readAltitude(SEALEVELPRESSURE_HPA);
            else data.altitude = 0.0;

            buffer[bufferIndex++] = data;

            if (bufferIndex >= BUFFER_SIZE) {
                dataFile.write((uint8_t*)buffer, sizeof(SensorData) * BUFFER_SIZE);
                bufferIndex = 0;
            }
        }

        // LED piscando D25
        if (millis() - ledFlashTime > 100) {
            ledFlashTime = millis();
            digitalWrite(LED_D25, !digitalRead(LED_D25));
        }

        // Verifica duração máxima
        if (millis() - startMillis >= RECORD_DURATION_MS) {
            Serial.println("Tempo de gravação atingido.");
            currentState = STATE_IDLE;
            break;
        }

        // Checa botão parar
        readButtons();
        if (currentState != STATE_RECORD_SD) break;

        vTaskDelay(1 / portTICK_PERIOD_MS);
    }

    // Grava o que sobrou no buffer
    if (bufferIndex > 0 && dataFile) {
        dataFile.write((uint8_t*)buffer, sizeof(SensorData) * bufferIndex);
        bufferIndex = 0;
    }

    if (dataFile) {
        dataFile.flush();
        dataFile.close();
        fileCreated = false;
        Serial.println("Gravação finalizada e arquivo fechado.");
    }

    digitalWrite(LED_D25, HIGH);
    digitalWrite(LED_D32, LOW);
    digitalWrite(LED_D33, LOW);
}

// ======================== SERIAL BIN ========================
void sendSerialBinary() {
    Serial.println("READY");
    unsigned long startMillis = millis();
    while (millis() - startMillis < RECORD_DURATION_MS && currentState == STATE_SERIAL_BIN) {
        SensorData data;
        data.timestamp = micros();

        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax,&ay,&az,&gx,&gy,&gz);
        data.mpu_accel[0] = ax; data.mpu_accel[1] = ay; data.mpu_accel[2] = az;
        data.mpu_gyro[0]  = gx; data.mpu_gyro[1]  = gy; data.mpu_gyro[2]  = gz;

        data.adxl_accel[0] = adxl.getX();
        data.adxl_accel[1] = adxl.getY();
        data.adxl_accel[2] = adxl.getZ();

        if (bmp.performReading()) data.altitude = bmp.readAltitude(SEALEVELPRESSURE_HPA);
        else data.altitude = 0.0;

        Serial.write((uint8_t*)&data, sizeof(SensorData));
        vTaskDelay(4 / portTICK_PERIOD_MS); // ~200 Hz
        readButtons();
    }
    currentState = STATE_IDLE;
    Serial.println("\n[Serial Binary] Transmissão finalizada.");
}

// ======================== MONITOR 5HZ ========================
void serialMonitor() {
    unsigned long startMillis = millis();
    while (millis() - startMillis < MONITOR_DURATION_MS && currentState == STATE_MONITOR_5HZ) {
        SensorData data;
        data.timestamp = micros();

        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax,&ay,&az,&gx,&gy,&gz);
        data.mpu_accel[0] = ax; data.mpu_accel[1] = ay; data.mpu_accel[2] = az;
        data.mpu_gyro[0]  = gx; data.mpu_gyro[1]  = gy; data.mpu_gyro[2]  = gz;

        data.adxl_accel[0] = adxl.getX();
        data.adxl_accel[1] = adxl.getY();
        data.adxl_accel[2] = adxl.getZ();

        if (bmp.performReading()) data.altitude = bmp.readAltitude(SEALEVELPRESSURE_HPA);
        else data.altitude = 0.0;

        Serial.print("MPU Accel: "); Serial.print(ax); Serial.print(", "); Serial.print(ay); Serial.print(", "); Serial.print(az);
        Serial.print(" | Gyro: "); Serial.print(gx); Serial.print(", "); Serial.print(gy); Serial.print(", "); Serial.println(gz);
        Serial.print("ADXL Accel: "); Serial.print(data.adxl_accel[0]); Serial.print(", "); Serial.print(data.adxl_accel[1]); Serial.print(", "); Serial.println(data.adxl_accel[2]);
        Serial.print("Altitude: "); Serial.println(data.altitude);
        Serial.println("-----------------------------------");

        vTaskDelay(200 / portTICK_PERIOD_MS); // 5 Hz
        readButtons();
    }
    currentState = STATE_IDLE;
}

// ======================== STATUS ========================
void printSystemStatus() {
    Serial.println("=== Sistema IDLE ===");
    Serial.print("MPU6050: ±8g / ±1000°/s | Status: OK\n");
    Serial.print("ADXL375: ±200g | Status: OK\n");
    Serial.print("BMP580: Status: OK | ODR 218Hz\n");
    Serial.print("SD: ");
    if (SD.begin(SD_CS)) Serial.println("OK");
    else Serial.println("Erro");
    Serial.print("Próximo arquivo: "); Serial.println("/DATA" + String(nextFileNumber) + ".BIN");
    Serial.println("-----------------------------------");
}
