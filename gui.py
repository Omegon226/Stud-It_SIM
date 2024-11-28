import streamlit as st
import tkinter as tk
import numpy as np
import pandas as pd
import requests
from pprint import pformat, pprint
import os


st.header("Подгрузка данных")
uploaded_files = st.file_uploader(
    "Подгрузите данные в csv формате (файл/множество файлов)",
    accept_multiple_files=True,
    type=['csv']
)

# Глобальная переменная названия файла с данными
file_name = None
if st.button("Отправить данные на сервис", key="K_1"):
    if len(uploaded_files) != 0:
        uploaded_files_to_request = []
        for uploaded_file in uploaded_files:
            uploaded_files_to_request += [("files", (uploaded_file.name, uploaded_file.getvalue(), "text/csv"))]

        result = requests.post(
            "http://127.0.0.1:8000/upload-files/",
            files=uploaded_files_to_request
        )

        if result.status_code == 200:
            st.write(f"Имеющиеся данные для обработки: {result.json()}")
        else:
            st.write(f"Что то пошло не так: {result.__dict__}")


st.header("Предобработка данных")
data_file_name = st.text_input("Введите полное название файла", key="11111")
if st.button("Обработать выбранный файл", key="K_2"):
    result = requests.post(
        "http://127.0.0.1:8000/transform-data/",
        json={
            "file_name": data_file_name
        }
    )

    if result.status_code == 200:
        st.write(f"Данные успешно были обработаны, теперь их размер = {result.json()} (байт)")
    else:
        st.write(f"Что то пошло не так: {result.__dict__}")


st.header("Выбор датасета")
datasets_names = requests.get("http://127.0.0.1:8000/datasets").json()
st.write(f"Доступные датасеты: {datasets_names}")
dataset_file_name = st.text_input("Введите полное название файла", key="1222221111")
if st.button("Обработать выбранный файл", key="K_3"):
    result = requests.post(
        "http://127.0.0.1:8000/set-dataset/",
        json={
            "dataset_name": dataset_file_name
        }
    )

    if result.status_code == 200:
        st.write(f"Датасет был успешно выбран")
    else:
        st.write(f"Что то пошло не так: {result.__dict__}")


st.header("Выбор модели")
models_names = requests.get("http://127.0.0.1:8000/models").json()
st.write(f"Доступные датасеты: {models_names}")
model_file_name = st.text_input("Введите полное название файла", key="54645645")
if st.button("Обработать выбранный файл", key="K_4"):
    result = requests.post(
        "http://127.0.0.1:8000/set-model/",
        json={
            "model_name": dataset_file_name
        }
    )

    if result.status_code == 200:
        st.write(f"Выбор модели был произведён успешно")
    else:
        st.write(f"Что то пошло не так: {result.__dict__}")


st.header("Создание предсказания на датасете")
if st.button("Сделать прогноз", key="K_5"):
    prediction = requests.get("http://127.0.0.1:8000/make-forecast").json()
    st.write(f"Результат предсказания")
    st.json(prediction)


st.header("Построение графика")
st.write("Доступные фичи: feature_1, feature_2, feature_3")
column_for_plot = st.selectbox(
    "Выберите название фичи для визуализации",
    ("feature_1", "feature_2", "feature_3"),
)
plot_type = st.selectbox(
    "Выберите тип визуализации",
    ("scatter", "line"),
)
if st.button("Создать визуализацию", key="K_6"):
    response = requests.post(
        "http://127.0.0.1:8000/make-plot",
        params={"col_name": column_for_plot, "plot_type": plot_type}
    )

    response = np.array(response.json())
    st.image(response, channels="RGB")

st.header("Информация об авторе")
if st.button("Получить информацию", key="K_1337"):
    info = requests.get("http://127.0.0.1:8000/get-service-data").json()
    st.write(f"{info}")

st.header("Инструкция")
if st.button("Вывести инструкцию", key="K_7"):
    def quit_me():
        print('quit')
        root.quit()
        #root.destroy()

    instruction = requests.get("http://127.0.0.1:8000/get-all-endpoints").json()
    for key in instruction.keys():
        instruction[key] = str(instruction[key]).replace("\n", "")
        instruction[key] = ' '.join(instruction[key].split())

    root = tk.Tk()
    service_text = tk.Text(root, height=50, width=300)
    service_text.pack()
    service_text.insert(1.0, pformat(instruction, compact=True, indent=2, width=1000))
    root.protocol("WM_DELETE_WINDOW", quit_me)
    root.geometry("800x500+100+100")
    root.title('Инструкция')
    root.mainloop()


if st.button("Вывести полную инструкцию", key="K_8"):
    os.startfile(r"doc\Система.docx")