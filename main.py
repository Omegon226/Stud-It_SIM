from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Response
import uvicorn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import StandardScaler
import dill
from catboost import CatBoostClassifier, Pool
import io
from typing import List


__version__ = 1.0
app = FastAPI()
# Глобальная переменная для pd.DataFrame, обычно в нём хранится датасет
df = None
# Глобальная переменная для ML моделей
model = None
# Глобальная переменнаяв которой хранится массив с последними предсказаниями
prediction = None


@app.post("/upload-files/")
async def upload_files(files: List[UploadFile] = File(...)):
  """
  Ручка для получения начальных данных.
  Данные подаются через данный эндпоинт в формате CSV. Во всех столбцах должны быть значение int или float.
  Всего в датасете должно быть 3 столбца, если их будет больше, то лишние будут отброшены.

  В датасете должны содеражтся колонки: feature_1, feature_2, feature_3

  :param files: - файлы в формате csv в котором есть колонки feature_1, feature_2, feature_3

  :return: - все файлы с данными (не датасетами!!!!) которые были успешно обработаны
  """
  try:
    for file in files:
      file_name = file.filename
      df = pd.read_csv(file.file)

      if len(df.columns) < 3:
        return HTTPException(status_code=400, detail="Недостаточно колонок данных")
      if "feature_1" not in df.columns:
        return HTTPException(status_code=400, detail="Нет колонки feature_1")
      if "feature_2" not in df.columns:
        return HTTPException(status_code=400, detail="Нет колонки feature_2")
      if "feature_3" not in df.columns:
        return HTTPException(status_code=400, detail="Нет колонки feature_3")

      df = df[["feature_1", "feature_2", "feature_3"]]
      df.to_csv(f"data/{file_name}")

    return os.listdir(r"C:\Users\skills\PycharmProjects\pythonProject\data")
  except Exception as error:
    return HTTPException(status_code=500, detail=str(error))


@app.post("/transform-data")
def transform_data(file_name: str = "test_data.csv"):
  """
  Ручка для трансформации данных. В данный эндпоинт передаётся название файла из папки data.
  Данные предобробатываются и сохраняются в datasets в формате pickle

  :param file_name: - название csv файла из ранее подгруженных файлов

  :return: - размер преобразованного датасета
  """
  global df
  try:
    scaler = StandardScaler()

    df = pd.read_csv(fr"C:\Users\skills\PycharmProjects\pythonProject\data\{file_name}")
    df = df.drop(columns=["Unnamed: 0"])

    df["feature_1"] = np.log(df["feature_1"])

    df_train_scaled = scaler.fit_transform(df.copy())
    df_train_scaled = pd.DataFrame(df_train_scaled, columns=df.columns)
    df = df_train_scaled.copy()

    del df_train_scaled

    with open(fr"C:\Users\skills\PycharmProjects\pythonProject\datasets\{file_name}.pickle", "wb") as file:
      pickle.dump(df, file)

    return os.path.getsize(fr"C:\Users\skills\PycharmProjects\pythonProject\datasets\{file_name}.pickle")
  except Exception as error:
    return HTTPException(status_code=500, detail=str(error))


@app.get("/datasets")
def datasets():
  """
  Ручка для получения доступных датасетов в сервисе

  :return: - доступные датасеты в ПЗУ сервиса
  """
  try:
    return os.listdir(r"C:\Users\skills\PycharmProjects\pythonProject\datasets\\")
  except Exception as error:
    return HTTPException(status_code=500, detail=str(error))


@app.post("/set-dataset")
def set_dataset(dataset_name: str = "test_data.csv.pickle"):
  """
  Ручка для подгрузки датасета из ПЗУ сервиса. Требуется указать название датасета, который требуется подгрузить

  :param dataset_name: - название датасета, которое хранится в ПЗУ сервиса (можно посмотреть в get datasets)

  :return: - подгруженный датасет в формате словаря из list
  """
  global df
  try:
    with open(fr"C:\Users\skills\PycharmProjects\pythonProject\datasets\{dataset_name}", "rb") as file:
      df = pickle.load(file)

    return df.to_dict(orient="list")
  except Exception as error:
    return HTTPException(status_code=500, detail=str(error))


@app.get("/models")
def get_models():
  """
  Ручка для получения доступных моделей в сервисе

  :return: - модели которые хранятся в ПЗУ сервиса
  """
  try:
    return os.listdir(r"C:\Users\skills\PycharmProjects\pythonProject\models\\")
  except Exception as error:
    return HTTPException(status_code=500, detail=str(error))


@app.post("/set-model")
def set_model(model_name: str = "CatBoost.cbm"):
  """
  Ручка для установки рабочей модели. 
  ВНИМАНИЕ, нужно писать не KNN, а полное название файла KNN.dill

  :param model_name: - название ML модели, котороя хранится в ПЗУ сервиса

  :return: - название установленной модели
  """
  global model
  try:
    if "KNN.dill":
      with open(r"C:\Users\skills\PycharmProjects\pythonProject\models\KNN.dill", "rb") as file:
        model = dill.load(file)
    elif "RandomForest.dill":
      with open(r"C:\Users\skills\PycharmProjects\pythonProject\models\RandomForest.dill", "rb") as file:
        model = dill.load(file)
    elif "CatBoost.cbm":
        model = CatBoostClassifier().load_model(r"C:\Users\skills\PycharmProjects\pythonProject\models\CatBoost.cbm")

    return model_name
  except Exception as error:
    return HTTPException(status_code=500, detail=str(error))


@app.get("/make-forecast")
def make_forecast():
  """
  Ручка для создания прогноза моделями
  ВНИМАНИЕ, перед запуском ручки требуется подгрузить датасет в ОЗУ сервис, а также модель

  :return: - массив предсказаний по установленным датасету и модели
  """
  global df, model, prediction
  try:
    if df is None:
      return HTTPException(status_code=400, detail="Датасет для прогнозирования не был подгружен")
    if model is None:
      return HTTPException(status_code=400, detail="Модуль для прогнозирования не была подгружена")

    prediction = model.predict(df)
    prediction = prediction.tolist()

    return prediction
  except Exception as error:
    return HTTPException(status_code=500, detail=str(error))


@app.post("/make-plot")
def make_plot(col_name: str, plot_type: str):
  """
  Ручка для построения графика зависимости между фичёй и прогнозом
  ВАНИМЕНИЕ в col_name могут передаваться только: feature_1, feature_2, feature_3

  :param col_name: - название фичи, которая будет визуализироваться на графике

  :return: - график в формате массива
  """
  global df, prediction
  try:
    if df is None:
      return HTTPException(status_code=400, detail="Датасет для прогнозирования не был подгружен")
    if prediction is None:
      return HTTPException(status_code=400, detail="Не были созданы предсказания")
    if col_name not in ["feature_1", "feature_2", "feature_3"]:
      return HTTPException(status_code=400, detail=f"Фичи {col_name} нет в данных")

    plot_df = df.copy()
    plot_df["prediction"] = prediction

    print(col_name)
    print(plot_type)

    fig, ax = plt.subplots(figsize=(2, 2))
    if plot_type == "scatter":
      plot_df.plot.scatter(x=col_name, y="prediction", ax=ax, c="prediction", cmap="viridis")
    if plot_type == "line":
      plot_df.plot.line(x=col_name, y="prediction", ax=ax)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data.tolist()
  except Exception as error:
    return HTTPException(status_code=500, detail=str(error))


@app.get("/get-all-endpoints")
def get_all_endpoints():
  """
    Ручка для получения справки о ручках

    :return: - документация ручек
    """
  try:
    return{
      "POST /upload-files": upload_files.__doc__,
      "POST /transform-data": transform_data.__doc__,
      "GET /datasets": datasets.__doc__,
      "POST /set-dataset": set_dataset.__doc__,
      "GET /models": get_models.__doc__,
      "POST /set-model": set_model.__doc__,
      "GET /make-forecast": make_forecast.__doc__,
      "POST /make-plot": make_plot.__doc__,
      "GET /get-all-endpoints": get_all_endpoints.__doc__,
      "GET /get-service-data": get_service_data.__doc__,
      "POST /make-plot-aboba": plot_data.__doc__
    }
  except Exception as error:
    return HTTPException(status_code=500, detail=str(error))


@app.get("/get-service-data")
def get_service_data():
  """
    Ручка для получения данных о сервисе

    :return: - информация о сервисе
  """
  try:
    return f"Версия: {__version__} , Разработчик: Смирнов Иван Михайлович , Компания: LABEAN"
  except Exception as error:
    return HTTPException(status_code=500, detail=str(error))


@app.post("/make-plot-aboba")
def plot_data(background_tasks: BackgroundTasks, col_name: str = "feature_1", plot_type: str = "scatter"):
  """
  Ручка для построения графика зависимости между фичёй и прогнозом
  ВАНИМЕНИЕ в col_name могут передаваться только: feature_1, feature_2, feature_3

  :param col_name: - название фичи, которая будет визуализироваться на графике

  :return: - график в формате массива
  """
  global df, prediction
  try:
    if df is None:
      return HTTPException(status_code=400, detail="Датасет для прогнозирования не был подгружен")
    if prediction is None:
      return HTTPException(status_code=400, detail="Не были созданы предсказания")
    if col_name not in ["feature_1", "feature_2", "feature_3"]:
      return HTTPException(status_code=400, detail=f"Фичи {col_name} нет в данных")

    plot_df = df.copy()
    plot_df["prediction"] = prediction

    plt.rcParams['figure.figsize'] = [7.50, 3.50]
    plt.rcParams['figure.autolayout'] = True
    if plot_type == "scatter":
      plot_df.plot.scatter(x=col_name, y="prediction", cmap="viridis")
    if plot_type == "line":
      plot_df.plot.line(x=col_name, y="prediction")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    buf_contents = img_buf.getvalue()
    background_tasks.add_task(img_buf.close)

    headers = {
      'Content-Disposition': 'inline; filename="out.png"'
    }
    return Response(buf_contents, headers=headers, media_type='image/png')
  except Exception as error:
    return HTTPException(status_code=500, detail=str(error))


if __name__ == "__main__":
  pass