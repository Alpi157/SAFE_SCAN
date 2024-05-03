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

class PassportProcessor:
    def __init__(self, model_path):
        # Загрузка модели YOLOv5, предварительно обученной для определенной страны
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.eval()  # Переключение модели в режим оценки

    def detect(self, image_path):
        # Загрузка изображения и его обработка через модель
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image. Check the file path.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразование в RGB
        results = self.model(img)  # Подача изображения в модель
        return results

    def extract_text(self, image_path):
        # Метод, который должен быть реализован в дочерних классах
        raise NotImplementedError

    def analyze_results(self, results):
        # Метод для анализа результатов детекции, специфичный для каждой страны
        raise NotImplementedError

    def extract_and_verify_passport_data(self, text):
        # Метод для извлечения и проверки данных паспорта на основе текста
        raise NotImplementedError


class AzerbaijanPassportProcessor(PassportProcessor):
    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image. Check the file path.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(img)
        return results

    def analyze_results(self, results):
        labels = results.names
        detected_classes = [labels[int(x[-1])] for x in results.xyxy[0]]
        required_elements = ['face', 'doc_quad', 'signature', 'detail1', 'detail2']
        missing_elements = [element for element in required_elements if element not in detected_classes]
        if len(missing_elements) == 0:
            return "Пройдено."
        else:
            missing_elements_str = ', '.join(missing_elements)
            return f"Скорее всего, паспорт не настоящий. Отсутствующие элементы: {missing_elements_str}"

    def extract_text(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image. Check the file path.")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
        thresh_img = cv2.adaptiveThreshold(morph_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
        custom_config = r'--oem 3 --psm 4'
        text = pytesseract.image_to_string(Image.fromarray(thresh_img), config=custom_config, lang='rus+eng')
        return text

    def extract_and_verify_passport_data(self, text):
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
        return {'passport_number': passport_number.group(0) if passport_number else "not found", 'expiry_status': expiry_status}

    def check_passport_in_database(self, passport_number):
        with open('passport_db.txt', 'r') as file:
            passport_list = file.read().strip().split(',')
            passport_set = set(passport_list)
        return passport_number in passport_set

    def get_metadata_using_exiftool(self, image_path):
        path_to_exiftool = r"C:\Users\arman\Downloads\exiftool-12.84\exiftool.exe"
        try:
            result = subprocess.run([path_to_exiftool, '-json', image_path], stdout=subprocess.PIPE, text=True, check=True)
            metadata = json.loads(result.stdout)
            return metadata[0] if metadata else None
        except subprocess.CalledProcessError as e:
            return {"error": f"Exiftool error: {e}"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {e}"}

    def analyze_editing_signals(self, metadata):
        responses = []
        if 'Software' in metadata and 'Adobe Photoshop' in metadata['Software']:
            responses.append(f"Файл был отредактирован в: {metadata['Software']}")
        if 'Software' in metadata and 'GIMP' in metadata['Software']:
            responses.append(f"Файл был отредактирован в: {metadata['Software']}")
        if 'ModifyDate' in metadata:
            responses.append(f"Дата последнего изменения файла: {metadata['ModifyDate']}")
        if 'HistorySoftwareAgent' in metadata and 'Adobe Photoshop' in metadata['HistorySoftwareAgent']:
            responses.append(f"Изменения были сделаны с использованием: {metadata['HistorySoftwareAgent']}")
        return "\n".join(responses) if responses else "Признаков редактирования с помощью известного ПО не обнаружено."




class EstoniaIDProcessor(PassportProcessor):
    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image. Check the file path.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(img)
        return results

    def analyze_results(self, results):
        labels = results.names
        detected_classes = [labels[int(x[-1])] for x in results.xyxy[0]]
        required_elements = ['face', 'doc_quad', 'signature', 'detail1', 'detail2']
        missing_elements = [element for element in required_elements if element not in detected_classes]
        if len(missing_elements) == 0:
            return "Пройдено."
        else:
            missing_elements_str = ', '.join(missing_elements)
            return f"Скорее всего, удостоверение не настоящее. Отсутствующие элементы: {missing_elements_str}"

    def extract_text(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image. Check the file path.")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
        thresh_img = cv2.adaptiveThreshold(morph_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
        custom_config = r'--oem 3 --psm 11'  # Изменено на 'psm 11' для лучшего анализа блока текста
        text = pytesseract.image_to_string(Image.fromarray(thresh_img), config=custom_config, lang='eng')
        return text

    def extract_and_verify_passport_data(self, text):
        # Поиск персонального кода, предположительно находящегося после даты рождения
        personal_code_match = re.search(r'\b\d{11}\b', text)

        # Поиск последней даты в тексте, которая предполагается как дата окончания действия
        expiry_dates = re.findall(r'(\d{2}\.\d{2}\.\d{4})', text)
        expiry_date_match = expiry_dates[-1] if expiry_dates else None

        expiry_status = "dates not found"
        if expiry_date_match:
            expiry_date = datetime.strptime(expiry_date_match, "%d.%m.%Y")
            if expiry_date < datetime.now():
                expiry_status = f"просрочено в {expiry_date.strftime('%d.%m.%Y')}"
            else:
                expiry_status = f"действительно до {expiry_date.strftime('%d.%m.%Y')}"

        return {
            'personal_code': personal_code_match.group(0) if personal_code_match else "not found",
            'expiry_status': expiry_status
        }

    def check_passport_in_database(self, id_number):
        with open('passport_db.txt', 'r') as file:
            passport_list = file.read().strip().split(',')
            passport_set = set(passport_list)
        return id_number in passport_set

    def get_metadata_using_exiftool(self, image_path):
        path_to_exiftool = r"C:\Users\arman\Downloads\exiftool-12.84\exiftool.exe"
        try:
            result = subprocess.run([path_to_exiftool, '-json', image_path], stdout=subprocess.PIPE, text=True, check=True)
            metadata = json.loads(result.stdout)
            return metadata[0] if metadata else None
        except subprocess.CalledProcessError as e:
            return {"error": f"Exiftool error: {e}"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {e}"}

    def analyze_editing_signals(self, metadata):
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
    return render_template('index2.html')

@app.route('/', methods=['POST'])
def upload_image():
    country = request.form['country']
    if 'file' not in request.files:
        return "Error: No file part."
    file = request.files['file']
    if file.filename == '':
        return "Error: No selected file."
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    processor = None
    if country == 'azerbaijan':
        processor = AzerbaijanPassportProcessor('C:\\Users\\arman\\PycharmProjects\\HakathonAPI\\yolov5\\runs\\train\\exp\\weights\\best.pt')
    elif country == 'estonia':
        processor = EstoniaIDProcessor('C:\\Users\\arman\\PycharmProjects\\HakathonAPI\\yolov5_est\\runs\\train\\exp\\weights\\best.pt')

    if processor:
        try:
            results = processor.detect(file_path)
            visual_check = processor.analyze_results(results)
            extracted_text = processor.extract_text(file_path)
            passport_data = processor.extract_and_verify_passport_data(extracted_text)
            if country == 'azerbaijan':
                passport_number = passport_data.get('passport_number', 'Not found')
                id_check = f"Номер паспорта: {passport_number}\n"
            elif country == 'estonia':
                personal_code = passport_data.get('personal_code', 'Not found')
                id_check = f"Персональный код: {personal_code}\n"

            passport_in_db = processor.check_passport_in_database(passport_data.get('personal_code', ''))
            metadata = processor.get_metadata_using_exiftool(file_path)
            editing_signals = processor.analyze_editing_signals(metadata) if metadata else "Metadata not found."

            response_message = f"Визуальная проверка: {visual_check}\n" \
                               f"{id_check}" \
                               f"Статус в базе данных: {'Присутствует' if passport_in_db else 'Отсутствует'}\n" \
                               f"Срок действия документа: {passport_data.get('expiry_status', 'Неизвестен')}\n" \
                               f"Анализ метаданных: {editing_signals}"

            os.remove(file_path)

            return response_message
        except Exception as e:
            return f"Error: {str(e)}"
    else:
        return "Error: Invalid country specified."



if __name__ == '__main__':
    app.run(debug=True)