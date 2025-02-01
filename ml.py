import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

image_path = 'test2.png'  # Замените на свой путь
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Уменьшение шума с помощью размытия
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Улучшенная бинаризация (метод Otsu)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Поиск контуров
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Фильтрация слишком маленьких областей (чтобы убрать шум)
filtered_contours = [c for c in contours if cv2.contourArea(c) > 50]

# Сортировка контуров слева направо
filtered_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[0])

# Извлечение и нормализация цифр
digits = []
for c in filtered_contours:
    x, y, w, h = cv2.boundingRect(c)
    digit = thresh[y:y + h, x:x + w]

    # Добавление отступов вокруг цифры (улучшает распознавание)
    padding = 10
    digit = cv2.copyMakeBorder(digit, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    # Изменение размера до 28x28
    digit = cv2.resize(digit, (28, 28))
    digits.append(digit)

# === ВИЗУАЛИЗАЦИЯ ВЫДЕЛЕННЫХ ЦИФР ===
plt.figure(figsize=(10, 5))
for i, digit in enumerate(digits):
    plt.subplot(1, len(digits), i + 1)
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
plt.show()


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация данных
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),  # Второй сверточный слой
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),  # Дополнительный слой для лучшего обучения
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели (увеличенное количество эпох)
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)


predictions = []
for digit in digits:
    digit = digit.astype('float32') / 255.0  # Нормализация пикселей
    digit = digit.reshape((1, 28, 28, 1))  # Приведение к формату для модели
    pred = model.predict(digit)
    predictions.append(np.argmax(pred))

print("Предсказанные цифры:", predictions)
