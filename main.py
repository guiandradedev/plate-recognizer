import numpy as np
import cv2
import os
import uuid
from fast_plate_ocr import LicensePlateRecognizer
from colors import Colors
from ultralytics import YOLO
import torch
import re

def load_yolo(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        Colors.error(f"Erro: Modelo não carregado, {e}")
        exit()

def detect(car_model, plate_model, plate_ocr_model, image_path, output_path, unique_id):
    img = cv2.imread(image_path)

    yolo_result = car_model(source=img, conf=0.5, classes=[2,3])

    result = yolo_result[0] # Como só tem uma imagem, pega o resultado 0

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
        index += 1
    cv2.imshow(image_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(len(result))


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


def run():
    car_model = load_yolo("yolo11s.pt")
    plate_model = load_yolo("plate-model.pt")
    # plate_ocr_model = LicensePlateRecognizer('global-plates-mobile-vit-v2-model')
    plate_ocr_model = LicensePlateRecognizer('cct-s-v1-global-model')

    image_path = "images_test/moto.jpeg"

    if not os.path.exists(image_path):
        Colors.error(f"Erro: Imagem não encontrada em {image_path}")
        exit()
    
    unique_id = str(uuid.uuid4())[:8]
    output_path = "output"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    detect(car_model, plate_model, plate_ocr_model, image_path, output_path, unique_id)

if __name__ == "__main__":
    run()