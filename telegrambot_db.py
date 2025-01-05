import sqlite3
import telebot
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Подключение к базе данных SQLite
conn = sqlite3.connect("face_recognition.db", check_same_thread=False)
cursor = conn.cursor()

# Создание таблицы для хранения данных (если еще не создана)
cursor.execute('''
CREATE TABLE IF NOT EXISTS recognized_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1
)
''')
conn.commit()

# Функция для обновления базы данных
def update_database(name):
    cursor.execute("SELECT count FROM recognized_faces WHERE name = ?", (name,))
    result = cursor.fetchone()
    if result:
        # Если имя уже есть в базе, увеличиваем счетчик
        cursor.execute("UPDATE recognized_faces SET count = count + 1 WHERE name = ?", (name,))
    else:
        # Если имени нет, добавляем его в базу
        cursor.execute("INSERT INTO recognized_faces (name, count) VALUES (?, ?)", (name, 1))
    conn.commit()

# Загрузка модели
cnn_model = load_model('face_recognition_model.h5')
data_directory = "/root/.cache/kagglehub/datasets/vasukipatel/face-recognition-dataset/versions/1/Original Images/Original Images/"

# Генератор изображений для предварительной обработки
image_gen = ImageDataGenerator()
train_data = image_gen.flow_from_directory(data_directory, target_size=(224, 224), batch_size=32)
class_names = list(train_data.class_indices.keys())

# Функция для предсказания
def predict_person(image_path, threshold=0.7):
    # Загрузка и преобразование изображения
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Прогнозирование
    predictions = cnn_model.predict(img_array, batch_size=32)

    # Получаем максимальную вероятность и имя предсказанного класса
    max_prob = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    if max_prob > threshold:
        update_database(predicted_class)  # Сохраняем данные в базе
        return f"Я думаю, это: {predicted_class} (Уверенность: {max_prob:.2f}). Пользователь сохранен в базе данных под именем {predicted_class}."
    else:
        update_database("Unknown")  # Сохраняем как "Unknown", если лицо не распознано
        return "Я не знаю этого человека. Пользователь сохранен в базе данных под именем Unknown."

# Инициализация бота
bot = telebot.TeleBot('7835023314:AAG3sTBMs9-cmiLG_xBEbfrtoj6TvdlL06Q')

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    # Сохранение фотографии
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open("received_photo.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    # Прогнозирование
    result = predict_person("received_photo.jpg")

    # Отправка результата пользователю
    bot.reply_to(message, result)

@bot.message_handler(content_types=['text'])
def handle_text(message):
    if message.text == "/photo":
        bot.send_message(message.from_user.id, "Привет, отправь мне фото для распознавания человека на нем.")

    elif message.text == "/database":
        # Получение данных из базы и отправка пользователю
        cursor.execute("SELECT name, count FROM recognized_faces")
        rows = cursor.fetchall()
        if rows:
            response = "Отсканированные люди:\n"
            for row in rows:
                response += f"{row[0]}: {row[1]} раз(а)\n"
            bot.send_message(message.from_user.id, response)
        else:
            bot.send_message(message.from_user.id, "База данных пуста.")
    else:
        bot.send_message(message.from_user.id, "Я тебя не понимаю. Выбери существующую команду.")

keep_alive() #запускаем flask-сервер в отдельном потоке.
bot.polling(non_stop=True, interval=0) #запуск бота
