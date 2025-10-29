# 🛰️ Coleta-MPU-BMP-ADXL – Aquisição e Análise de Movimento Multissensor

Esta pasta reúne os códigos e dados utilizados na **aquisição e análise de movimento utilizando múltiplos sensores inerciais e barométricos** conectados ao **ESP32**.  
O objetivo é capturar com alta precisão o comportamento dinâmico de um corpo em movimento, realizando fusão sensorial e reconstrução da trajetória por meio de técnicas de processamento digital de sinais e orientação espacial.

---

## ⚙️ Arquitetura do Sistema

O sistema é composto por três sensores principais e um módulo de armazenamento, todos integrados ao **ESP32**:

| Componente | Função principal | Interface | Observações |
|-------------|------------------|------------|--------------|
| **MPU-6050** | Acelerômetro + Giroscópio com DMP | I²C | Fornece aceleração linear e velocidade angular.|
| **ADXL (ex: ADXL345)** | Acelerômetro externo auxiliar | I²C | Usado para validação cruzada e comparação com o MPU-6050, aumentando a confiabilidade da medição. |
| **BMP (ex: BMP180 / BMP280)** | Sensor barométrico | I²C | Mede pressão atmosférica e temperatura para estimar variações de altitude. |
| **Cartão SD** | Armazenamento local de dados | SPI | Armazena os dados brutos em formato binário (`.bin`), permitindo coletas longas e de alta taxa sem perda de informação. |

A taxa de amostragem é mantida em **200 Hz**, garantindo boa resolução temporal para estudos dinâmicos de curta duração.

---

## 🧩 Descrição dos Arquivos

| Arquivo | Linguagem | Descrição |
|----------|------------|-----------|
| **`Coleta-ESP32.ino`** | C++ (Arduino) | Código embarcado no ESP32. Realiza leitura simultânea dos sensores via I²C, armazena os dados no cartão SD (formato `.bin`), e opcionalmente transmite os valores pela serial. O código controla a taxa de amostragem em **200 Hz** e sincroniza as leituras para minimizar jitter. |
| **`Coleta-Serial-CSV.py`** | Python | Recebe os dados transmitidos pela serial e grava em arquivo `.csv`. Ideal para coletas curtas e depuração em tempo real. Inclui verificação de integridade de pacotes e registro de tempo. |
| **`TCC-BIN-CSV.py`** | Python | Converte os arquivos `.bin` gravados no SD para `.csv`, permitindo análise direta em Python. A estrutura de dados segue o formato `[tempo, ax, ay, az, gx, gy, gz, pressão, temperatura, ax_aux, ay_aux, az_aux]`. |
| **`Grafico-verificação.py`** | Python | Script leve que permite visualizar rapidamente os sinais coletados (aceleração, giroscópio, pressão, etc.) para confirmar se a coleta foi bem-sucedida. |
| **`Análise-de-movimento.py`** | Python | Realiza a análise avançada dos dados aplicando técnicas de processamento de sinais e fusão sensorial. Este é o script principal de análise. |

---

## 🧠 Principais Técnicas de Processamento

O código **`Análise-de-movimento.py`** utiliza uma série de algoritmos e métodos para reconstruir o movimento e corrigir erros inerentes aos sensores:

### 🔹 1. Pré-processamento
- Remoção de offset e normalização de unidades.  
- Correção de drift de giroscópio.  
- Interpolação cúbica dos dados para garantir espaçamento temporal uniforme.  

### 🔹 2. Filtragem
- **Filtros FIR e IIR** (Butterworth, FIR HP, etc.) para remoção de ruído.  
- Comparação entre abordagens **causais** e **não causais** (`lfilter` vs `filtfilt`).  

### 🔹 3. Fusão Sensorial e Orientação
- Implementação de algoritmos **AHRS (Attitude and Heading Reference System)** como **Madgwick** e **Mahony**.  
- Filtro complementar para fusão entre aceleração e giroscópio.  
- Representação da orientação em **quatérnios** e conversão para ângulos de Euler (roll, pitch, yaw).  

### 🔹 4. Correção de Orientação
- Ajuste de quadros de referência utilizando as estimativas de atitude.  
- Projeção da aceleração linear no eixo global para eliminação do componente gravitacional.  

### 🔹 5. Integração Numérica
- Cálculo de velocidade e deslocamento via **integração trapezoidal**.  
- Aplicação de técnicas de **remoção de tendência** para minimizar erro acumulado por ZUTP.  

### 🔹 6. Visualização e Interpretação
- Geração de gráficos no domínio do tempo e frequência.  
- Comparação entre sensores (MPU × ADXL).  
- Análise de altitude.

---

## 🧰 Estrutura dos Dados

Os dados armazenados (em `.bin` e `.csv`) seguem o formato:

| Coluna | Unidade | Descrição |
|---------|----------|-----------|
| **Tempo** | segundos | Marca temporal de cada amostra. |
| **Ax, Ay, Az** | m/s² | Aceleração medida pelo MPU-6050. |
| **Gx, Gy, Gz** | °/s | Velocidade angular do MPU-6050. |
| **Altitude** | M | Leitura do BMP. |
| **Ax_aux, Ay_aux, Az_aux** | m/s² | Aceleração medida pelo ADXL. |

---

## ⚙️ Fluxo de Operação

1. **Coleta:** o ESP32 lê todos os sensores a 200 Hz e grava em formato binário (`.bin`) no cartão SD.  
2. **Conversão:** o script `TCC-BIN-CSV.py` transforma o arquivo em `.csv`.  
3. **Verificação:** `Grafico-verificação.py` é usado para inspecionar rapidamente os sinais coletados.  
4. **Análise:** `Análise-de-movimento.py` aplica filtros, fusões, correções e integrações, gerando gráficos e resultados quantitativos.

---

## 📊 Exemplos de Resultados

- Gráficos de aceleração e velocidade angular.  
- Curvas de altitude em função da pressão.  
- Comparações entre sensores MPU e ADXL.  
- Estimativas de deslocamento e orientação espacial.

*(Os exemplos e gráficos gerados podem ser encontrados nos notebooks e scripts de análise.)*

---

## 💡 Observações Importantes

- O sistema foi projetado para capturar movimentos rápidos de curta duração (ex: trajetória de um foguete de garrafa PET).  
- Pequenas variações de sincronismo entre sensores são corrigidas via interpolação temporal.  
- O formato binário foi escolhido para minimizar latência e maximizar desempenho na escrita em SD.  
- O código foi testado em diferentes taxas de amostragem, sendo **200 Hz** o ponto ótimo entre estabilidade e resolução.  

---

## 👤 Autor

**Luis Felipe Pereira Ramos**  
Técnico em Automação Industrial – IFMT  
Desenvolvido como parte do Trabalho de Conclusão de Curso (TCC).
