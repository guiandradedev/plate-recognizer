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

    file = './data/plates/data.yaml'
    model = YOLO("yolo11s.pt")

    model.train(
        data=file,
        
        # Configurações SEGURAS para RTX 4060 Ti 8GB
        workers=4,       # Ainda menos workers
        device=[0],      # GPU única
        batch=8,         # Batch ainda menor
        imgsz=416,       # Imagem menor
        
        # Configurações de treinamento
        epochs=300,
        patience=50,
        optimizer='SGD',
        lr0=0.01,
        
        # Data augmentation leve
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        # fliplr=0.3,
        # flipud=0.3
        
        # Configurações de projeto
        project='runs/plate/yolo-safe',
        name='safe-4060ti-nano',
        exist_ok=True,
        resume=False,
        
        # Configurações simples
        cos_lr=False,
        warmup_epochs=1,
        weight_decay=0.0001,
        
        # Salvamento
        save_period=25,
        val=True,
        plots=False,  # Desliga plots para economizar memória
        verbose=True
    )