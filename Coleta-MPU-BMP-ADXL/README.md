# 🛰️ Coleta-MPU-BMP-ADXL – Aquisição e Análise de Dados Multissensor

Esta pasta reúne os códigos e arquivos de teste utilizados para a **coleta, conversão e análise dos dados de múltiplos sensores** conectados ao ESP32.  
O sistema combina informações do **MPU-6050**, **BMP (sensor barométrico)** e **ADXL (acelerômetro externo)** para permitir uma caracterização mais completa do movimento e do comportamento dinâmico do corpo de prova.

---

## 📜 Arquivos disponíveis

| Arquivo | Linguagem | Descrição |
|----------|------------|-----------|
| **`Coleta-ESP32.ino`** | C++ (Arduino) | Código principal embarcado no ESP32. Realiza a leitura dos sensores MPU-6050, BMP e ADXL via comunicação **I2C**, grava os dados em cartão SD via **SPI**, e pode enviar as leituras em tempo real pela **serial** ou exibir gráficos simples. Opera a uma taxa de atualização de **200 Hz**. |
| **`Coleta-Serial-CSV.py`** | Python | Recebe os dados transmitidos pela serial (modo de transmissão direta do ESP32) e salva em um arquivo `.csv` para posterior análise. |
| **`Grafico-verificação.py`** | Python | Script simples e rápido para verificar a integridade das coletas gravadas. Permite uma visualização inicial dos dados antes da análise completa. |
| **`TCC-BIN-CSV.py`** | Python | Converte arquivos `.bin` (coletas salvas em formato binário pelo ESP32) para `.csv`, tornando-os compatíveis com os scripts de análise. |
| **`Análise-de-movimento.py`** | Python | Realiza a análise completa do movimento do corpo, aplicando técnicas avançadas de **processamento de sinais e fusão sensorial**: filtros FIR e IIR, interpolação cúbica, algoritmos AHRS, filtros complementares, uso de quatérnions, correção de orientação e integração numérica (método dos trapézios), entre outros. |
| **Arquivos `.csv`** | Dados | Coletas de teste nomeadas conforme a situação experimental (ex: sensor parado, movimento linear, rotação, etc.). Servem como base para validação e comparação dos algoritmos. |

---

## 🧩 Fluxo geral de funcionamento

1. **Coleta:** o ESP32 lê os dados brutos dos sensores via I2C e os grava no cartão SD (formato `.bin`) ou envia diretamente pela serial.  
2. **Conversão:** o script `TCC-BIN-CSV.py` converte os arquivos binários em `.csv`.  
3. **Verificação:** o script `Grafico-verificação.py` é usado para inspecionar rapidamente os dados coletados.  
4. **Análise completa:** o `Análise-de-movimento.py` aplica os filtros, fusões e integrações para reconstruir a trajetória e estimar o comportamento dinâmico.  

---

## ⚙️ Sensores e Comunicação

| Sensor | Função | Protocolo |
|--------|---------|-----------|
| **MPU-6050** | Acelerômetro + Giroscópio | I2C |
| **BMP (Barômetro)** | Medição de pressão e altitude | I2C |
| **ADXL** | Acelerômetro externo auxiliar | I2C |
| **Cartão SD** | Armazenamento de dados | SPI |

---

## 💡 Observações

- A taxa de amostragem de **200 Hz** foi escolhida para equilibrar precisão temporal e estabilidade de gravação.  
- Os algoritmos de fusão (AHRS, filtros complementares e quatérnions) são fundamentais para corrigir deriva e alinhar referências.  
- Todos os scripts Python foram escritos com foco na clareza e modularidade, permitindo fácil adaptação a novos sensores e formatos de dados.

---

## 📊 Próximos passos

- Comparar o desempenho dos diferentes métodos de fusão sensorial.  
- Integrar o módulo GPS para análise de trajetória absoluta.  
- Validar os resultados experimentais com medições reais em campo.  
- Implementar visualização 3D interativa da orientação do corpo.

---

## 👤 Autor

**Luis Felipe Pereira Ramos**  
Desenvolvido como parte do Trabalho de Conclusão de Curso (TCC).  
Técnico em Automação Industrial – Instituto Federal do Mato Grosso (IFMT)
