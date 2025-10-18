# ⚙️ ESP32-DMP – Leitura e Processamento de Dados da DMP do MPU-6050

Esta pasta contém os códigos utilizados para testar e analisar o funcionamento da **DMP (Digital Motion Processor)** do sensor **MPU-6050** em conjunto com o **ESP32**, além dos scripts em **Python** para recepção e análise dos dados.

---

## 📜 Arquivos disponíveis

| Arquivo | Linguagem | Descrição |
|----------|------------|-----------|
| **`ESP32-DMP-PLOTER.cpp`** | C++ | Código que realiza a leitura dos valores processados pela DMP (aceleração e velocidade angular) e exibe as informações no monitor serial. |
| **`ESP32-DMP-ENVIO.cpp`** | C++ | Versão otimizada que envia os dados da DMP em formato binário pela serial, permitindo maior taxa de transmissão. |
| **`Receber-dados-DMP.py`** | Python | Script que recebe os dados binários enviados pelo ESP32 e salva em um arquivo `.csv` para análise posterior. |
| **`Processamento-DMP.py`** | Python | Lê os arquivos `.csv`, aplica escalas aos dados, gera gráficos e realiza análises de comportamento do sensor. |
| **`Dados_MPU_Parado.csv`** | Dados | Registro de uma coleta realizada com o sensor em repouso, utilizada para análise de ruído e estabilidade. |
| **`Dados_MPU_com_troca_de_orientação.csv`** | Dados | Registro de uma coleta com o sensor sendo movimentado, utilizada para observar o comportamento dinâmico da DMP. |

---

## 🧩 Funcionamento geral

1. **Captura:** o ESP32 lê os dados da DMP do MPU-6050 (aceleração, giroscópio, quaternios, ângulos de Euloer ....).  
2. **Transmissão:** os dados são enviados pela porta serial, em formato texto ou binário.  
3. **Recepção:** o script em Python recebe os dados e armazena em um arquivo `.csv`.  
4. **Processamento:** outro script realiza a leitura dos arquivos, aplica escalas e exibe gráficos para análise visual.

---

## 💡 Observações

- As escalas de leitura (±8 g e ±1000 °/s) foram configuradas para adequar o sensor à faixa de medição esperada.  
- Pequenos atrasos iniciais (nas primeiras amostras) podem ocorrer devido ao processamento interno da DMP.  
- Os códigos Python foram escritos de forma simples e didática para facilitar a compreensão do fluxo completo.

---

## 🔧 Próximos passos

- Implementar gravação automática em cartão SD.  
- Sincronizar os dados da DMP com leituras de outros sensores (ex: GPS).  
- Explorar métodos de filtragem e fusão sensorial para reconstrução da trajetória.

---

## 👤 Autor

**Luis Felipe Pereira Ramos**  
Desenvolvido como parte do Trabalho de Conclusão de Curso (TCC).
