# Projeto YOLO Placas

Este projeto utiliza o modelo YOLO para detecção de carros e placas veiculares, além de OCR para leitura das placas.

## Estrutura principal
- `main.py`: Script principal para detecção de carros e placas em imagens.
- `train.py`: Script para treinamento do modelo YOLO.
- `validate.py`: Script para validação do modelo treinado.
- `colors.py`: Utilitários de cores para visualização.
- `plate.py`: Funções relacionadas à manipulação de placas.
- `requirements.txt`: Dependências do projeto.
- `yolo11s.pt` e `plate-model.pt`: Pesos dos modelos YOLO.
- `data/plates/`: Dados de treinamento, validação e teste.
- `images_test/`: Imagens para teste rápido.
- `output/`: Resultados gerados pelas detecções.

## Como usar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute o script principal:
   ```bash
   python main.py
   ```

## Observação
- Os scripts de treinamento e validação requerem configuração dos caminhos dos dados e modelos.
- O projeto utiliza Ultralytics YOLO e um OCR para placas veiculares.

## OCR de Placas
Para realizar a leitura das placas, é necessário instalar a biblioteca [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) conforme o ambiente físico (Windows, Linux, etc). Siga as instruções do repositório para garantir o funcionamento correto do OCR.
