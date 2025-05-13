import requests
from bs4 import BeautifulSoup
import csv
import time
import random
import re

count = 1
# Список для хранения данных
apartments = []

# Список user-agентов для имитации запроса от браузера
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
]

# Функция для выбора случайного user-agent
def get_random_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

# Функция для парсинга строки, например, "2-комн. квартира, 57,5 м², 6/17 этаж"
def parse_apartment_details(detail_str):
    rooms = 0
    square = 'Не указана'
    floor = 'Не указан'
    home_floor = 'Не указан'

    # Ищем количество комнат
    room_match = re.search(r'(\d+)-комн\.', detail_str)
    if room_match:
        rooms = room_match.group(1)
    
    # Ищем площадь квартиры (может быть вещественным числом с запятой)
    square_match = re.search(r'(\d+,\d+|\d+)\s?м²', detail_str)
    if square_match:
        square = float(square_match.group(1).replace(',', '.'))

    # Ищем этажи (например, "6/17 этаж")
    floor_match = re.search(r'(\d+)/(\d+)\s?этаж', detail_str)
    if floor_match:
        floor = floor_match.group(1)  # этаж квартиры
        home_floor = floor_match.group(2)  # общее количество этажей в доме

    return rooms, square, floor, home_floor


while count <= 100:
    url = f"https://krasnodar.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&p={count}&region=4820"
    
    # Выполнение GET-запроса
    response = requests.get(url, headers=get_random_headers())
    
    # Проверка успешности запроса
    if response.status_code == 200:
        print(f"Успех! Данные со страницы {count}.")
        soup = BeautifulSoup(response.text, 'html.parser')

        # Находим все объявления о квартирах
        ads = soup.find_all('div', class_='_93444fe79c--card--ibP42 _93444fe79c--wide--gEKNN')

        # Извлекаем данные из каждого объявления
        for ad in ads:
            # Заголовок (Название квартиры) - изменено на "Квартира"
            title = "Квартира"
            
            # Извлечение цены из атрибута data-mark="MainPrice"
            price = ad.find('span', {'data-mark': 'MainPrice'})
            if price:
                price = price.get_text(strip=True)
            else:
                price = 'Цена не указана'
            
            # Местоположение
            location = ad.find('div', class_='_93444fe79c--labels--L8WyJ').get_text(strip=True) if ad.find('div', class_='_93444fe79c--labels--L8WyJ') else 'Нет местоположения'
            
            # Детали квартиры (например, "2-комн. квартира, 57,5 м², 6/17 этаж")
            details = ad.find('div', class_='_93444fe79c--row--kEHOK').get_text(strip=True) if ad.find('div', class_='_93444fe79c--row--kEHOK') else ''
            
            # Парсим детали
            rooms, square, floor, home_floor = parse_apartment_details(details)
            
            # Извлечение названия ЖК
            jkh_name = ad.find('a', class_='_93444fe79c--jk--dIktL')
            if jkh_name:
                jkh_name = jkh_name.get_text(strip=True)
            else:
                jkh_name = 'ЖК не указан'

            # Добавляем все данные в список
            apartments.append({
                'title': title,
                'jkh': jkh_name,  # ЖК добавлен после title
                'location': location,
                'rooms': rooms,
                'square': square,
                'floor': floor,
                'home-floor': home_floor,
                'price': price,  # Цена добавляется в конец
            })
# Случайная пауза между запросами
            time.sleep(random.uniform(1, 3))  # Пауза 1-3 секунды
    else:
        print(f"Ошибка при запросе:{response.status_code}")
        break
    
    count += 1

# Запись собранных данных в CSV файл
with open('krasnodar_apartments.csv', mode='w', newline='', encoding='utf-8') as file:
    # Определяем заголовки для CSV
    fieldnames = ['title', 'jkh', 'location', 'rooms', 'square', 'floor', 'home-floor', 'price']  # ЖК после title
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    writer.writeheader()  # Запись заголовков в CSV
    for apartment in apartments:
        writer.writerow(apartment)  # Запись каждой квартиры

print("Данные успешно записаны в krasnodar_apartments.csv")