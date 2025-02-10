# Используем официальный образ Python
FROM python:3.10.16

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл use_caesar_cipher.py из директории src в контейнер
COPY src/use_caesar_cipher.py /app/

# Указываем команду по умолчанию, которая будет выполняться при запуске контейнера
CMD ["python", "use_caesar_cipher.py"]