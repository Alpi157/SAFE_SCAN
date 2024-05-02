import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import torch
import cv2
from PIL import Image
import numpy as np
import re
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
import os
import subprocess
import json
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\arman\PycharmProjects\HakathonAPI\yolov5\runs\train\exp\weights\best.pt')
model.eval()


def detect(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение. Проверьте путь к файлу.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img)
    return results


def analyze_results(results):
    labels = results.names
    detected_classes = [labels[int(x[-1])] for x in results.xyxy[0]]
    required_elements = ['face', 'doc_quad', 'signature', 'detail1', 'detail2']
    missing_elements = [element for element in required_elements if element not in detected_classes]

    if len(missing_elements) == 0:
        return "Паспорт прошёл визуальную проверку."
    else:
        missing_elements_str = ', '.join(missing_elements)
        return f"Скорее всего, паспорт не настоящий. Отсутствующие элементы: {missing_elements_str}"


def extract_text(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение. Проверьте путь к файлу.")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)

    thresh_img = cv2.adaptiveThreshold(morph_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)

    custom_config = r'--oem 3 --psm 4'
    text = pytesseract.image_to_string(Image.fromarray(thresh_img), config=custom_config, lang='rus+eng')
    return text

def extract_and_verify_passport_data(text):

    passport_number = re.search(r'C\d+', text)
    dates = re.findall(r'(\d{2}\.\d{2}\.\d{4})', text)
    expiry_status = "dates not found"
    if dates:
        birth_date = datetime.strptime(dates[0], "%d.%m.%Y")
        expiry_date = datetime.strptime(dates[-1], "%d.%m.%Y")
        if expiry_date < datetime.now():
            expiry_status = f"просрочен в {expiry_date.strftime('%d.%m.%Y')}"
        else:
            expiry_status = f"годен до {expiry_date.strftime('%d.%m.%Y')}"
    passport_info = {
        'passport_number': passport_number.group(0) if passport_number else "not found",
        'expiry_status': expiry_status
    }
    return passport_info


def get_metadata_using_exiftool(image_path):
    path_to_exiftool = r"C:\Users\arman\Downloads\exiftool-12.84\exiftool.exe"
    try:
        result = subprocess.run([path_to_exiftool, '-json', image_path], stdout=subprocess.PIPE, text=True, check=True)
        metadata = json.loads(result.stdout)
        return metadata[0] if metadata else None
    except subprocess.CalledProcessError as e:
        return {"error": f"Exiftool error: {e}"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}"}


def analyze_editing_signals(metadata):
    responses = []
    if 'Software' in metadata and 'Adobe Photoshop' in metadata['Software']:
        responses.append(f"Файл был отредактирован в: {metadata['Software']}")
    if 'ModifyDate' in metadata:
        responses.append(f"Дата последнего изменения файла: {metadata['ModifyDate']}")
    if 'HistorySoftwareAgent' in metadata and 'Adobe Photoshop' in metadata['HistorySoftwareAgent']:
        responses.append(f"Изменения были сделаны с использованием: {metadata['HistorySoftwareAgent']}")
    return "\n".join(responses) if responses else "Признаков редактирования с помощью известного ПО не обнаружено."



@app.route('/', methods=['GET'])
def upload_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "Ошибка: Не найден файл."
    file = request.files['file']
    if file.filename == '':
        return "Ошибка: Файл не выбран."
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        detection_results = detect(file_path)
        visual_check = analyze_results(detection_results)
        extracted_text = extract_text(file_path)
        passport_data = extract_and_verify_passport_data(extracted_text)
        metadata = get_metadata_using_exiftool(file_path)
        editing_signals = analyze_editing_signals(metadata) if not isinstance(metadata, dict) or "error" not in metadata else metadata.get("error", "")

        response_message = (f"Визуальная проверка: {visual_check}\n"
                            f"Номер паспорта: {passport_data.get('passport_number', 'Не найден')}\n"
                            f"Срок действия паспорта: {passport_data.get('expiry_status', 'Неизвестен')}\n"
                            f"Анализ метаданных: {editing_signals}")
        return response_message
    except Exception as e:
        return f"Ошибка: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)