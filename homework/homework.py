# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import glob
import os
import gzip
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


"""
Paso 1.
Realice la limpieza de los datasets:
- Renombre la columna "default payment next month" a "default".
- Remueva la columna "ID".
- Elimine los registros con informacion no disponible.
- Para la columna EDUCATION, valores > 4 indican niveles superiores
  de educación, agrupe estos valores en la categoría "others".
"""


def clean_datasets(filepath):
    df = pd.read_csv(filepath, compression='zip')
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df = df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: x if x <= 4 else 4)

    df = df.dropna()
    return df


"""
Paso 2.
Divida los datasets en x_train, y_train, x_test, y_test.

"""

def split_datasets(df):
    x = df.drop(columns=["default"])
    y = df["default"]
    return x, y


"""
Paso 3.
Cree un pipeline para el modelo de clasificación. Este pipeline debe
contener las siguientes capas:
- Transforma las variables categoricas usando el método
  one-hot-encoding.
- Descompone la matriz de entrada usando componentes principales.
  El pca usa todas las componentes.
- Escala la matriz de entrada al intervalo [0, 1].
- Selecciona las K columnas mas relevantes de la matrix de entrada.
- Ajusta una red neuronal tipo MLP.
"""

def create_pipeline(x_train):
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = [col for col in x_train.columns if col not in categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='passthrough'
    )


    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA()),
        ('selectk', SelectKBest(score_func=f_classif)),
        ('mlp', MLPClassifier(max_iter=15000, random_state=42))
    ])
    
    return pipeline


"""
Paso 4.
Optimice los hiperparametros del pipeline usando validación cruzada.
Use 10 splits para la validación cruzada. Use la función de precision
balanceada para medir la precisión del modelo.
"""

def optimize_hyperparameters(pipeline, x_train, y_train):
    param_grid = {
        'pca__n_components': [None],
        'selectk__k': [20],
        'mlp__hidden_layer_sizes': [(50, 30, 40, 60)],
        'mlp__alpha': [0.28],
        'mlp__learning_rate_init': [0.001]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,

        scoring='balanced_accuracy',
        refit=True
    )

    grid.fit(x_train, y_train)
    return grid

"""
Paso 5.
Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

"""

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(model, f)



"""
Paso 6.
Calcule las metricas de precision, precision balanceada, recall,
y f1-score para los conjuntos de entrenamiento y prueba.
Guardelas en el archivo files/output/metrics.json. Cada fila
del archivo es un diccionario con las metricas de un modelo.
Este diccionario tiene un campo para indicar si es el conjunto
de entrenamiento o prueba. Por ejemplo:

{'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
{'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}

"""

def calculate_metrics(y_true, y_pred, dataset_type):
    return {
        "type": "metrics",
        'dataset': dataset_type,
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
    }

"""
Paso 7.
Calcule las matrices de confusion para los conjuntos de entrenamiento y
prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
del archivo es un diccionario con las metricas de un modelo.
de entrenamiento o prueba. Por ejemplo:

{'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
{'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
"""


def calculate_confusion_matrix(y_true, y_pred, dataset_type):
    cm = confusion_matrix(y_true, y_pred)
    return {
        'type': 'cm_matrix',
        'dataset': dataset_type,
        'true_0': {
            'predicted_0': int(cm[0, 0]),
            'predicted_1': int(cm[0, 1])
        },
        'true_1': {
            'predicted_0': int(cm[1, 0]),
            'predicted_1': int(cm[1, 1])
        }
    }

test_path = "files/input/test_data.csv.zip"
train_path = "files/input/train_data.csv.zip"
test_data_pd = clean_datasets(test_path)
train_data_pd = clean_datasets(train_path)
x_train, y_train = split_datasets(train_data_pd)
x_test, y_test = split_datasets(test_data_pd)
pipeline = create_pipeline(x_train)
optimized_model = optimize_hyperparameters(pipeline, x_train, y_train)
save_model(optimized_model, "files/models/model.pkl.gz")
y_train_pred = optimized_model.predict(x_train)
y_test_pred = optimized_model.predict(x_test)
train_metrics = calculate_metrics(y_train, y_train_pred, "train")
test_metrics = calculate_metrics(y_test, y_test_pred, "test")
train_cm = calculate_confusion_matrix(y_train, y_train_pred, "train")
test_cm = calculate_confusion_matrix(y_test, y_test_pred, "test")

os.makedirs("files/output", exist_ok=True)

all_metrics = [train_metrics, test_metrics, train_cm, test_cm]
pd.DataFrame(all_metrics).to_json("files/output/metrics.json", orient="records", lines=True)




