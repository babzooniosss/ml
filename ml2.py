

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Загрузка предобученной модели для распознавания цифр
model = load_model('mnist_model.h5')

# Загрузка изображения
image = cv2.imread('./Digits.png', cv2.IMREAD_GRAYSCALE)

# Адаптивная бинаризация
binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Морфологические операции для устранения шумов
kernel = np.ones((3, 3), np.uint8)
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Поиск контуров
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Визуализация контуров
output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
cv2.imshow("Contours", output_image)
cv2.waitKey(0)

# Вывод количества найденных контуров
print(f"Найдено контуров: {len(contours)}")


def preprocess_image(image_path):
    # Загрузка изображения и преобразование в черно-белый формат
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Применение пороговой обработки для получения бинарного изображения
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def segment_digits(image):
    # Нахождение контуров цифр
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Сортировка контуров слева направо
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    digits = []
    for contour in contours:
        # Получение ограничивающего прямоугольника для каждого контура
        x, y, w, h = cv2.boundingRect(contour)
        # Вырезание цифры из изображения
        digit = image[y:y+h, x:x+w]
        # Изменение размера цифры до 28x28 (как в MNIST)
        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        digit = img_to_array(digit)
        digits.append(digit)
    return np.array(digits)

def recognize_digits(digits):
    # Нормализация изображений
    digits = digits.astype('float32') / 255.0
    # Распознавание цифр с помощью нейронной сети
    predictions = model.predict(digits)
    recognized_digits = [np.argmax(prediction) for prediction in predictions]
    return recognized_digits

def main(image_path):
    # Предобработка изображения
    binary_image = preprocess_image(image_path)
    # Сегментация цифр
    digits = segment_digits(binary_image)
    # Распознавание цифр
    recognized_digits = recognize_digits(digits)
    print("Распознанные цифры:", recognized_digits)

if __name__ == "__main__":
    image_path = './Digits.png' 
    main(image_path)
