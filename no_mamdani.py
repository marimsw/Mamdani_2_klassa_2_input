import os
from ultralytics import YOLO

# Путь к файлам изображений
image_path = '/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/YOLOv8_and_vesa_18_plus/66.jpg'

# Проверка существования файла
if not os.path.exists(image_path):
    print(f"Файл не найден: {image_path}")
else:
    # Загрузка моделей с весами
    model1 = YOLO('/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/YOLOv8_and_vesa_18_plus/best.pt')
    model2 = YOLO('/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/YOLOv8_and_vesa_18_plus/best1.pt')

    # Функция для обработки изображения
    def detect_objects(model, image_path):
        # Выполнение детекции на изображении
        results = model(image_path)

        # Словарь для хранения достоверностей
        confidences = {}

        # Обработка результатов
        for result in results:
            boxes = result.boxes  # Получаем объект с детекциями
            for box in boxes:
                conf = box.conf[0]  # Достоверность
                class_id = int(box.cls[0])  # ID класса
                class_name = model.names[class_id]  # Название класса

                # Сохраняем максимальную достоверность для каждого класса
                if class_name not in confidences:
                    confidences[class_name] = conf
                else:
                    confidences[class_name] = max(confidences[class_name], conf)

        return confidences

    # Примеры использования
    if __name__ == "__main__":
        # Получаем достоверности для всех классов из обеих моделей
        confidences1 = detect_objects(model1, image_path)
        confidences2 = detect_objects(model2, image_path)

        # Объединяем результаты
        all_confidences = {**confidences1, **confidences2}

        # Выводим достоверности
        print("Достоверности классов:")
        for class_name, confidence in all_confidences.items():
            print(f"{class_name}: {confidence:.2f}")

        # Проверяем, превышает ли хотя бы одна достоверность 0.7
        if any(confidence > 0.7 for confidence in all_confidences.values()):
            print("Контент обнаженный, запрещенный")
        else:
            print("Контент разрешенный")
