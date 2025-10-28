# 🧠 Projeto TCC – Análise e Captura de Movimento com ESP32 e MPU6050

Este repositório reúne todos os códigos, scripts e dados desenvolvidos durante o Trabalho de Conclusão de Curso (TCC) de **Luis Felipe Pereira Ramos**, com foco na **aquisição e processamento de dados de sensores inerciais** (MPU-6050) utilizando o **ESP32** e scripts de análise em **Python**.

O objetivo principal do projeto é **capturar, registrar e analisar o movimento de um sistema físico**, aplicando técnicas de filtragem e integração numérica para obter deslocamento, velocidade e comportamento dinâmico do corpo em estudo.

---

## 📂 Estrutura do Repositório

O repositório está organizado em pastas temáticas.  
Cada pasta contém seus próprios códigos e um arquivo `README.md` explicando em detalhes os arquivos contidos.

| Pasta | Descrição |
|-------|------------|
| **ESP32-DMP/** | Códigos relacionados à leitura de dados da DMP (Digital Motion Processor) do MPU-6050 utilizando o ESP32, além dos scripts Python para receber, salvar e analisar os dados. |
| *Coleta-MPU-BMP-ADXL/* | Outras pastas poderão ser adicionadas explorando filtros digitais, GPS, fusão sensorial, entre outros módulos do sistema. |

---

## ⚙️ Tecnologias Utilizadas

- **ESP32** – microcontrolador responsável pela coleta de dados dos sensores.  
- **MPU-6050** – sensor de aceleração e giroscópio com DMP integrada.  
- **Python** – usado para comunicação serial, armazenamento de dados e processamento numérico.  
- **Bibliotecas principais:**  
  - `Wire.h`, `MPU6050_6Axis_MotionApps20.h` (C++)  
  - `pyserial`, `numpy`, `matplotlib`, `pandas`, `scipy` (Python)

---

## 🚀 Como navegar pelo projeto

1. Acesse a pasta de interesse (ex: [`ESP32-DMP`](./ESP32-DMP)).  
2. Leia o `README.md` dentro da pasta para entender os arquivos e como executar cada parte.  
3. Caso deseje reproduzir os testes, siga as instruções específicas contidas em cada pasta.

---

## 👨‍💻 Autor

**Luis Felipe Pereira Ramos**  
Técnico em Automação Industrial pelo IFMT  
Projeto desenvolvido como parte do Trabalho de Conclusão de Curso (TCC).

📧 [luis.felipe.ramos@unemat.br](mailto:luis.felipe.ramos@unemat.br)  
📘 [LinkedIn](www.linkedin.com/in/luis-felipe-pereira-ramos-357843346)  
🐙 [GitHub](https://github.com/Felipovisky-Eng)

---

## 🧾 Licença

Este projeto é de uso acadêmico e livre para consulta.  
O uso ou modificação dos códigos é permitido mediante **citação do autor e do repositório original**.
