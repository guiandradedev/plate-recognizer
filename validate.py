import os
from ultralytics import YOLO
import torch

def print_cuda():
    print('GPU AVAILABLE:', torch.cuda.is_available())

    # If CUDA is available, print more detailed information
    if torch.cuda.is_available():
        # Print details for each GPU
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Memory Free: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / 1024**3:.2f} GB")

if __name__ == '__main__':
    print_cuda()

    # Caminho para o arquivo de dados
    data_file = './data/plates/data.yaml'

    # Caminho para o modelo treinado (ajuste conforme necessário)
    model_path = 'plate-model.pt'

    # Verificar se o modelo existe
    if not os.path.exists(model_path):
        print(f"Modelo não encontrado em {model_path}. Verifique o caminho.")
        exit(1)

    # Carregar o modelo treinado
    model = YOLO(model_path)

    # Executar validação
    print("Iniciando validação...")
    results = model.val(
        data=data_file,
        device=[0],  # Usar GPU
        batch=8,     # Mesmo batch size do treinamento
        imgsz=416,   # Mesmo tamanho de imagem
        save_json=True,  # Salvar resultados em JSON
        save_txt=True,   # Salvar predições em TXT
        plots=True,   # Gerar plots de validação
        verbose=True
    )

    # Imprimir métricas principais
    print("\nMétricas de Validação:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precisão: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")

    print("Validação concluída!")