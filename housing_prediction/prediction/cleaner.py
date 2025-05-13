import csv

def clean_price_in_csv(input_file, output_file):
    """
    Читает CSV-файл, удаляет знак рубля из столбца 'price',
    преобразует цену в числовой формат и записывает результат в новый файл.

    Args:
        input_file (str): Путь к входному CSV-файлу.
        output_file (str): Путь к выходному CSV-файлу.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
                open(output_file, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames  # Получаем имена столбцов
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()  # Записываем заголовки столбцов

            for row in reader:
                price_str = row['price']
                if price_str: # Проверяем, что цена не пустая
                    price_str = price_str.replace(" ", "").replace("₽", "").replace(",", ".")  # Убираем пробелы, ₽ и заменяем запятую на точку
                    try:
                        price = float(price_str) # Преобразуем в float
                    except ValueError:
                        price = None  # Если преобразование не удалось,  price будет None
                else:
                    price = None # Если цена изначально не указана, оставляем None

                row['price'] = price # Записываем очищенную цену

                writer.writerow(row)

        print(f"Файл {input_file} обработан и сохранен в {output_file}")

    except FileNotFoundError:
        print(f"Ошибка: Файл {input_file} не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Пример использования:
input_csv_file = 'krasnodar_apartments.csv'  # Замените на имя вашего входного файла
output_csv_file = 'krasnodar_apartments_cleaned.csv'  # Имя выходного файла
clean_price_in_csv(input_csv_file, output_csv_file)