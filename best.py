import os
from collections import defaultdict
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm

REAL_DATA_PATH = "data/Продажи.xlsx"  # Определение пути к файлу с реальными данными о продажах.
PAST_PREDICTIONS_FOLDER = "predictions полгода до"  # Определение пути к папке с прошлыми прогнозами.
BEST_MODEL_FOR_DEPARTMENTS_PATH = "data/best_results_my_mape.xlsx"  # Определение пути к файлу с лучшими моделями для каждого товара.
FUTURE_PREDICTIONS_FOLDER = "predictions"  # Определение пути к папке с будущими прогнозами.
BEST_FUTURE_PREDICTIONS_PATH = "data/future_best_preds2.xlsx"  # Определение пути к файлу с лучшими прогнозами на будущее.

CALCULATED_METRICS = {  # Создание словаря с метриками для вычисления ошибок.
    "MAE": mean_absolute_error,  # Добавление функции средней абсолютной ошибки в словарь.
    # "MAPE": mean_absolute_percentage_error 
}

def transform_data():
    df = pd.read_excel(REAL_DATA_PATH)
    df = df.fillna(0)
    return df

# Определение функции для получения прошлых прогнозов и связанных с ними данных.
def get_past_data(required_cols):
    pred_paths = glob(f"{PAST_PREDICTIONS_FOLDER}/*.xlsx")  # Получение путей ко всем файлам с прошлыми прогнозами.
    pred_names = [os.path.basename(path).split(".")[0] for path in pred_paths]  # Получение имен файлов без расширений.
    pred_dfs = [pd.read_excel(path)[required_cols] for path in pred_paths]  # Чтение данных из файлов и извлечение необходимых столбцов.

    return pred_names, pred_dfs  # Возврат имен и DataFrame прошлых прогнозов.

# Определение функции для расчета лучшей модели для каждого отдела.
def calc_best_method_for_department(departments, df, past_preds_names, past_preds_dfs):
    best_preds = []

    for department in tqdm(departments):
        y_true = list(df.loc[df["Отдел"] == department].values[0][1:])
        true_sum = sum(y_true)

        cur_metrics = defaultdict(list)
        cur_preds = []
        cur_preds_sum = []
        custom_metric = []

        for pred_df in past_preds_dfs:
            y_pred = list(pred_df.loc[pred_df["Отдел"] == department].values[0][1:])
            if any(pred < 0 for pred in y_pred):  # Пропускаем модели с отрицательными прогнозами
                continue

            cur_preds.append(y_pred)
            cur_preds_sum.append(sum(y_pred))
            # Вычисляем среднюю ошибку для каждого месяца
            errors = [abs(true - pred) for true, pred in zip(y_true, y_pred)]
            cur_metrics['MAE'].append(round(np.mean(errors), 2))

        # Выбираем модель с наименьшим средним MAE
        idx_min_metric = np.argmin([np.mean(errors) for errors in cur_metrics.values()])

        # Формируем текущий лучший прогноз
        cur_best = (
            [department] + cur_preds[idx_min_metric] + [past_preds_names[idx_min_metric]]
        )
        cur_best += (
            [true_sum]
            + [cur_preds_sum[idx_min_metric]]
            + [np.mean(cur_metrics['MAE'])]
        )

        # Добавляем вычисленные метрики ошибок
        for metric_name in CALCULATED_METRICS.keys():
            cur_best.append(cur_metrics[metric_name][idx_min_metric])

        best_preds.append(cur_best)

    return pd.DataFrame(
        best_preds,
        columns=list(df.columns)
        + ["ModelType", "true", "pred", "mean_MAE"]
        + list(CALCULATED_METRICS.keys()),
    )
def calc_my_mape(df):
    df["my_MAPE"] = 0  # Создание столбца "my_MAPE" и заполнение его нулями.
    df.loc[df["true"] != 0, "my_MAPE"] = round(
        abs(df["true"] - df["pred"]) / df["true"], 2
    )  # Вычисление MAPE для ненулевых реальных продаж.

    df["1 - my_MAPE"] = 1 - df["my_MAPE"]  # Вычисление 1 - MAPE.
    df[">= 90%"] = 0  # Создание столбца ">= 90%" и заполнение его нулями.
    df.loc[df["1 - my_MAPE"] >= 0.9, ">= 90%"] = 1  # Установка значения 1, если 1 - MAPE >= 0.7.

    return df  # Возврат DataFrame с добавленными столбцами.

def get_future_data():
    pred_paths = glob(f"{FUTURE_PREDICTIONS_FOLDER}/*.xlsx")  # Получение путей к файлам с будущими прогнозами.
    pred_names = [os.path.basename(path).split(".")[0] for path in pred_paths]  # Извлечение имен файлов.
    
    # Создание словаря, где ключи - имена прогнозов, а значения - DataFrame с данными о прогнозах.
    pred_dfs = {name: pd.read_excel(path) for name, path in zip(pred_names, pred_paths)}

    return pred_dfs  # Возврат словаря с будущими прогнозами.

def get_future_preds_for_best_model(best_model_for_department, future_preds_dfs, department_only_future):
    future_preds = []  # Создание пустого списка для будущих прогнозов.

    # Цикл по лучшим моделям для товаров.
    for _, row in tqdm(best_model_for_department.iterrows()):
        department, model = row["Отдел"], row["ModelType"]  # Получение текущего отдела и его лучшей модели.
        cur_df = future_preds_dfs[model]  # Получение DataFrame с прогнозами для текущей модели.
        future_pred = list(cur_df.loc[cur_df["Отдел"] == department].values[0][1:])  # Получение прогнозов для текущего отдела.

        future_preds.append([department] + future_pred + [model])  # Добавление прогнозов в список.


    # Создаем DataFrame с будущими прогнозами.
    return pd.DataFrame(
        future_preds,
        columns=["Отдел"]  # Столбец с идентификаторами отдела.
        + list(future_preds_dfs[list(future_preds_dfs.keys())[0]].columns[1:])  # Столбцы с прогнозами.
        + ["ModelType"],  # Столбец с именами моделей.
    )
if __name__ == "__main__":
    df = transform_data()
    print(df)
    past_preds_names, past_preds_dfs = get_past_data(list(df.columns))
    print(past_preds_dfs[0])

    # Оставляем только те товары, для которых есть прогнозы
    # department = list(past_preds_dfs[0]["item"].unique())
    # df = df[df["item"].isin(items)]
    department = list(df["Отдел"].unique())
    department_with_past = list(past_preds_dfs[0]["Отдел"].unique())
    df = df[df["Отдел"].isin(department_with_past)]
    department_only_future = list(set(department) - set(department_with_past))
    department = list(df["Отдел"].unique())

    # Для каждого товара вычисляем лучшую модель
    best_model_for_department = calc_best_method_for_department(
        department, df, past_preds_names, past_preds_dfs
    )
    print(best_model_for_department)

    # Подсчитываем собственное MAPE для лучшего метода
    best_model_for_department = calc_my_mape(best_model_for_department)
    print(best_model_for_department)

    print(best_model_for_department[">= 90%"].value_counts())

    # Сохраняем лучший метод с метриками для каждого отдела
    best_model_for_department.to_excel(BEST_MODEL_FOR_DEPARTMENTS_PATH, index=False)

    # подгружаем таблицы с предсказаниями каждого метода на будущее
    future_preds_dfs = get_future_data()

    # Составляем новую таблицу, где для каждого отдела берем предсказания на будущее из лучшего метода
    best_future_preds_for_department = get_future_preds_for_best_model(
        best_model_for_department, future_preds_dfs, department_only_future
    )

    # Сохраняем
    best_future_preds_for_department.to_excel(BEST_FUTURE_PREDICTIONS_PATH, index=False)
if __name__ == "__main__":
    # Если скрипт запускается напрямую:

    # Получаем данные о продажах и трансформируем их
    df = transform_data()
    print(df)

    # Получаем данные о прошлых прогнозах
    past_preds_names, past_preds_dfs = get_past_data(list(df.columns))
    print(past_preds_dfs[0])

    # Оставляем только те отделы, для которых есть прошлые прогнозы
    department = list(df["Отдел"].unique())
    department_with_past = list(past_preds_dfs[0]["Отдел"].unique())
    df = df[df["Отдел"].isin(department_with_past)]
    department_only_future = list(set(department) - set(department_with_past))
    department = list(df["Отдел"].unique())

    # Для каждого отдела вычисляем лучшую модель
    best_model_for_department = calc_best_method_for_department(
        department, df, past_preds_names, past_preds_dfs
    )
    print(best_model_for_department)

    # Подсчитываем собственное MAPE для лучшего метода
    best_model_for_department = calc_my_mape(best_model_for_department)
    print(best_model_for_department)

    # Печатаем количество отделов с MAPE >= 90%
    print(best_model_for_department[">= 90%"].value_counts())

    # Сохраняем лучший метод с метриками для каждого отдела в CSV-файл
    best_model_for_department.to_excel(BEST_MODEL_FOR_DEPARTMENTS_PATH, index=False)

    # Получаем данные о будущих прогнозах
    future_preds_dfs = get_future_data()

    # Получаем будущие прогнозы для лучшей модели каждого отдела
    best_future_preds_for_department = get_future_preds_for_best_model(
        best_model_for_department, future_preds_dfs, department_only_future
    )

    # Сохраняем будущие прогнозы в CSV-файл
    best_future_preds_for_department.to_excel(BEST_FUTURE_PREDICTIONS_PATH, index=False)