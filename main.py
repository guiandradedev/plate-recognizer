import numpy as np
import cv2
import os
import uuid
from fast_plate_ocr import LicensePlateRecognizer
from colors import Colors
from ultralytics import YOLO
import torch
import re
import matplotlib.pyplot as plt

def load_yolo(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        Colors.error(f"Erro: Modelo não carregado, {e}")
        exit()

def detect(car_model, plate_model, plate_ocr_model, color_model, image_path, output_path, unique_id):
    img = cv2.imread(image_path)

    yolo_result = car_model(source=img, conf=0.3, classes=[2,3])

    result = yolo_result[0] # Como só tem uma imagem, pega o resultado 0
    print(image_path, result, result.boxes)

    if len(result.boxes) <= 0: 
        Colors.error("Erro: Nenhum carro detectado na imagem")
        return
    
    # Se passou tem pelo menos um carro na imagem
    index = 0
    for car_bbox in result.boxes:
        print(car_bbox.conf, car_bbox.cls)
        # Converte as coordenadas do carro de tensor para um map
        x1, y1, x2, y2 = map(int, car_bbox.xyxy[0].tolist())

        conf = car_bbox.conf.item()
        class_id = int(car_bbox.cls.item()) # Sempre carro

        img_class = "carro" if class_id == 2 else "moto"

        text = f"{img_class} {index} - {(conf * 100):.2f}%"
        cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)

        car_cropped = img[y1:y2, x1:x2]

        detect_plate(plate_model, plate_ocr_model, car_cropped, index, img_class)
        car_color = classify_color(color_model, car_cropped)
        if car_color:
            print(f"Cor do {img_class} {index}: ", car_color)
            color_text = f"cor: {car_color}"
            cv2.putText(img, color_text, (x2 - 100, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
        index += 1

    img_name = image_path.split("/")[-1]
    cv2.imwrite(os.path.join(output_path, f"result-{unique_id}-{img_name}"), img)
    # cv2.imshow(image_path, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(len(result))

    return img


def show_images_side_by_side(images, titles=None, figsize=(15, 5), save_path=None):
    """Recebe uma lista de imagens no formato OpenCV (BGR) e salva um grid lado a lado usando matplotlib (RGB).

    Se save_path for None, a figura será exibida com plt.show(). Caso contrário, será salva no caminho informado.
    """
    if not images:
        print("Nenhuma imagem para exibir.")
        return

    n = len(images)
    fig = plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        # Converte BGR->RGB
        if img is None:
            rgb = None
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax = plt.subplot(1, n, i+1)
        if rgb is None:
            ax.text(0.5, 0.5, 'Imagem vazia', horizontalalignment='center')
        else:
            ax.imshow(rgb)
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i])
    plt.tight_layout()
    if save_path:
        # Cria diretório se necessário
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Grid salvo em: {save_path}")
    else:
        plt.show()

def classify_color(color_model, car_image):
    yolo_result = color_model(source=car_image, conf=0.2)
    
    result = yolo_result[0]

    # print(result)

    # if len(result.boxes) <= 0: 
    #     Colors.error("Erro: Nenhuma cor detectada na imagem do carro")
    #     return None
    
    # # Pega a cor com maior confiança
    # best_color = None
    # best_conf = 0.0
    # for color_bbox in result.boxes:
    #     conf = color_bbox.conf.item()
    #     class_id = int(color_bbox.cls.item())
    #     if conf > best_conf:
    #         best_conf = conf
    #         best_color = color_model.names[class_id]

    best_color = color_model.names[result.probs.top1]
    
    return best_color


def detect_plate(plate_model, plate_ocr_model,car_image, index, type_class):
    yolo_result = plate_model(source=car_image, conf=0.2)
    
    result = yolo_result[0]

    if len(result.boxes) <= 0: 
        Colors.error(f"Erro: Nenhuma placa detectada na imagem {index}")
        return
    
    for plate_bbox in result.boxes:
        # Converte as coordenadas da placa de tensor para um map
        x1, y1, x2, y2 = map(int, plate_bbox.xyxy[0].tolist())

        conf = plate_bbox.conf.item()
        # class_id = int(car_bbox.cls.item()) # Sempre placa

        plate_cropped = car_image[y1:y2, x1:x2]
        cv2.imwrite(f"output/plate-{index}.png", plate_cropped)
        plate = convert_plate_to_string(plate_ocr_model, plate_cropped, type_class)

        print(f"Placa detectada no carro {index}: ", plate)

        text = f"placa {index} - {(conf * 100):.2f}%"
        text2 = f"{plate}"
        position1 = (x1, y1-20)
        position2 = (x1, y1)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        color = (255, 0,0)
        line_type = cv2.LINE_AA
        cv2.putText(car_image, text, position1, font_face, font_scale, color, font_thickness, line_type)
        cv2.putText(car_image, text2, position2, font_face, font_scale, color, font_thickness, line_type)
        cv2.rectangle(car_image, (x1,y1), (x2,y2), (255,0,0), 2)

def convert_plate_to_string(plate_ocr_model, plate_image, type_class):
    # https://blog.dp6.com.br/regex-o-guia-essencial-das-express%C3%B5es-regulares-2fc1df38a481

    ocr_result = plate_ocr_model.run(plate_image)
    
    if not ocr_result or len(ocr_result) == 0:
        return ""

    plate = ocr_result[0].replace("_", "").strip()
    print(plate)

    plate = plate.upper()
    if type_class == "carro":
        plate_filtered = re.search(r'[A-Z]{3}[0-9][A-Z0-9][0-9]{2,3}', plate)
    else:
        plate_filtered = re.search(r'[A-Z]{3}[0-9]{4}', plate)

    return plate_filtered.group(0) if plate_filtered else ""


def run_many():
    car_model = load_yolo("yolo11s.pt")
    plate_model = load_yolo("plate-model.pt")
    color_model = load_yolo("color-classifier.pt")
    # plate_ocr_model = LicensePlateRecognizer('global-plates-mobile-vit-v2-model')
    plate_ocr_model = LicensePlateRecognizer('cct-s-v1-global-model')

    unique_id = str(uuid.uuid4())[:8]
    output_path = "output"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    min_images = 1
    max_images = 16
    images = []
    titles = []
    for i in range(min_images,max_images):
        image_path = f"images_test/image{i}.png"

        if not os.path.exists(image_path):
            Colors.error(f"Aviso: Imagem não encontrada em {image_path}, pulando.")
            continue

        print(f"\nProcessando imagem: {image_path}")
        # Executa detecção e obtém a imagem anotada (OpenCV BGR)
        img = detect(car_model, plate_model, plate_ocr_model, color_model, image_path, output_path, unique_id)
        if img is not None:
            images.append(img)
            titles.append(os.path.basename(image_path))

    # Se houver imagens processadas, mostra todas lado a lado
    if images:
        # Ajusta figsize dependendo do número de imagens
        width_per_image = 5
        figsize = (min(6 * len(images), 30), 6)
        save_path = os.path.join(output_path, f"grid-{unique_id}.png")
        show_images_side_by_side(images, titles=titles, figsize=figsize, save_path=save_path)
    else:
        print("Nenhuma imagem processada para exibir.")


def run():
    car_model = load_yolo("yolo11x.pt")
    plate_model = load_yolo("plate-model.pt")
    color_model = load_yolo("color-classifier.pt")
    # plate_ocr_model = LicensePlateRecognizer('global-plates-mobile-vit-v2-model')
    plate_ocr_model = LicensePlateRecognizer('cct-s-v1-global-model')

    image_path = "images_test/image7.png"

    if not os.path.exists(image_path):
        Colors.error(f"Erro: Imagem não encontrada em {image_path}")
        exit()
    
    unique_id = str(uuid.uuid4())[:8]
    output_path = "output"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    detect(car_model, plate_model, plate_ocr_model, color_model, image_path, output_path, unique_id)


if __name__ == "__main__":
    run_many()