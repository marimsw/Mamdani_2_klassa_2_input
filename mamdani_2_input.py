import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

def detect_objects(model, image_path, target_class):
    """
    Детекция объектов на изображении.

    Args:
        model (YOLO): Модель детекции объектов.
        image_path (str): Путь к изображению.
        target_class (str): Класс объекта, который нужно детектировать.

    Returns:
        float: Достоверность детекции объекта.
    """
    try:
        results = model(image_path)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name == target_class:
                    return conf.item()  # возвращаем доверие как число
        return 0.0
    except Exception as e:
        print(f"Ошибка детекции объектов: {e}")
        return 0.0

def fuzzy_membership_functions(x):
    """
    Определение членства для категорий: низкое, среднее и высокое.

    Args:
        x (numpy.array): Входные данные.
        
    Returns:
        tuple: Низкие, средние и высокие значения принадлежности.
    """
    low = fuzz.trimf(x, [0, 0, 0.5])
    medium = fuzz.trimf(x, [0, 0.5, 1])
    high = fuzz.trimf(x, [0.5, 1, 1])
    
    return low, medium, high

def fuzzy_rules(input1, input2):
    """
    Применение нечетких правил.

    Args:
        input1 (float): Первый входной параметр.
        input2 (float): Второй входной параметр.

    Returns:
        tuple: Выходные значения по правилам.
    """
    x = np.arange(0, 1.1, 0.1)  # Определяем диапазон
    low1, medium1, high1 = fuzzy_membership_functions(x)
    low2, medium2, high2 = fuzzy_membership_functions(x)

    # Получаем значения принадлежности
    input1_low = fuzz.interp_membership(x, low1, input1)
    input1_medium = fuzz.interp_membership(x, medium1, input1)
    input1_high = fuzz.interp_membership(x, high1, input1)

    input2_low = fuzz.interp_membership(x, low2, input2)
    input2_medium = fuzz.interp_membership(x, medium2, input2)
    input2_high = fuzz.interp_membership(x, high2, input2)

    # Применяем правила
    output_low = np.fmax(input1_low, input2_high)
    output_medium = np.fmin(input1_medium, input2_medium)
    output_high = np.fmax(input1_high, input2_medium)

    return output_low, output_medium, output_high

def defuzzification(output):
    """
    Дефазификация выходных значений.

    Args:
        output (tuple): Выходные значения по нечетким правилам.

    Returns:
        float: Дефазифицированное значение.
    """
    low, medium, high = output
    x = np.arange(0, 1.1, 0.1)  # Определяем диапазон
    aggregated = np.fmax(low, np.fmax(medium, high))

    # Преобразуем aggregated в массив, если это не так
    if np.isscalar(aggregated):
        aggregated = np.array([aggregated] * len(x))  # Создаем массив с одинаковыми значениями

    # Проверка согласованности размеров
    if len(x) != len(aggregated):
        print(f"Ошибка: длины x ({len(x)}) и aggregated ({len(aggregated)}) не совпадают.")
        return None  # Возвращаем None, если размеры не совпадают

    # Используем 'centroid' для дефазификации
    result = fuzz.defuzz(x, aggregated, 'centroid')
    return result


def plot_memberships(input1, input2, output):
    """
    Построение графиков степеней принадлежности.

    Args:
        input1 (float): Первый входной параметр.
        input2 (float): Второй входной параметр.
        output (tuple): Выходные значения по нечетким правилам.
    """
    x = np.arange(0, 1.1, 0.1)
    low1, medium1, high1 = fuzzy_membership_functions(x)
    low2, medium2, high2 = fuzzy_membership_functions(x)

    output_low, output_medium, output_high = output

    plt.figure(figsize=(12, 8))

    # График входной переменной 1
    plt.subplot(3, 2, 1)
    plt.title("Вход 1: 'boobs' Достоверность")
    plt.plot(x, low1, label='Низкое', color='blue')
    plt.plot(x, medium1, label='Среднее', color='green')
    plt.plot(x, high1, label='Высокое', color='red')
    plt.axvline(input1, color='black', linestyle='--', label='Достоверность входа 1')
    plt.legend()

    # График входной переменной 2
    plt.subplot(3, 2, 2)
    plt.title("Вход 2: 'naked female genitals' Достоверность")
    plt.plot(x, low2, label='Низкое', color='blue')
    plt.plot(x, medium2, label='Среднее', color='green')
    plt.plot(x, high2, label='Высокое', color='red')
    plt.axvline(input2, color='black', linestyle='--', label='Достоверность входа 2')
    plt.legend()

    # График выходных переменных
    plt.subplot(3, 1, 2)
    plt.title("Выход: Результаты нечеткой логики")
    plt.fill_between(x, output_low, color='blue', alpha=0.1, label='Низкое')
    plt.fill_between(x, output_medium, color='green', alpha=0.1, label='Среднее')
    plt.fill_between(x, output_high, color='red', alpha=0.1, label='Высокое')
    plt.axhline(0, color='black', lw=0.5)
    plt.axhline(1, color='black', lw=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main(image_path):
    """
    Главная функция, которая осуществляет нечеткую импликацию Мамдани.

    Args:
        image_path (str): Путь к изображению.
    """
    # Загрузка моделей с весами
    model1 = YOLO('/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/women_boobs_pisia/best.pt')
    model2 = YOLO('/content/drive/MyDrive/Rabota/Алгоритмы  Мамдани _от 18_11_24/women_boobs_pisia/naked_fem_genitals_weights.pt')

    # Детекция объектов на изображении
    input1 = detect_objects(model1, image_path, "boobs")
    input2 = detect_objects(model2, image_path, "naked female genitals")

    print(f"Объект 'boobs' обнаружен с достоверностью {input1:.2f}")
    print(f"Объект 'naked female genitals' обнаружен с достоверностью {input2:.2f}")

    # Проверяем, что хотя бы один объект обнаружен с достоверностью больше 0.7
    if (input1 > 0.07 and input2 == 0) or (input2 > 0.07 and input1 == 0) or (input1 > 0.07 and input2 > 0.7):
        output = fuzzy_rules(input1, input2)
        result = defuzzification(output)
        if result is not None:  # Проверка на успешную дефазификацию
            print(f"Результат нечеткой импликации: {result:.2f}")
            plot_memberships(input1, input2, output)
    else:
        print("Объекты не обнаружены с достаточной достоверностью.")

if __name__ == "__main__":
    # Путь к файлам изображений
    image_path = input('Введите путь до изображения: ')

    # Проверка существования файла
    # Проверка существования файла
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
    else:
        main(image_path)


