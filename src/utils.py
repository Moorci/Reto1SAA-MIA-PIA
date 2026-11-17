"""Funciones compartidas del proyecto (carga de datos, nulos, variables nuevas...)."""
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"
MODELS = RAIZ / "models"
RESULTADOS = RAIZ / "resultados"

# Columnas que entran al modelo (datos ya procesados en 01_eda.ipynb)
COLS_NUMERICAS = [
    "age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "contactado_antes", "contactos_totales",
]
COLS_CATEGORICAS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]

# MLflow: guardamos los experimentos en una base SQLite en la raíz del proyecto
MLFLOW_URI = f"sqlite:///{RAIZ / 'mlflow.db'}"
EXPERIMENTO = "reto1-deposito-banco"

# Columnas categóricas con pocos nulos que imputamos con la moda de train
COLS_MODA = ["job", "marital", "housing", "loan"]


def cargar_datos_raw(nombre_archivo: str, sep: str = ";") -> pd.DataFrame:
    """Carga un CSV de data/raw y lo devuelve como DataFrame.

    Args:
        nombre_archivo: nombre del fichero dentro de data/raw.
        sep: separador del CSV (el de Bank Marketing usa ";").
    """
    return pd.read_csv(RAW / nombre_archivo, sep=sep)


def unknown_a_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia del DataFrame con la categoría "unknown" convertida a NaN.

    En Bank Marketing los nulos vienen codificados como el texto "unknown".
    """
    return df.replace("unknown", pd.NA).copy()


def crear_variables_nuevas(df: pd.DataFrame) -> pd.DataFrame:
    """Añade las variables nuevas del proyecto y limpia el valor centinela de pdays.

    - contactado_antes: 1 si el cliente fue contactado en una campaña anterior
      (pdays distinto de 999), 0 si no.
    - pdays: el valor 999 (= no contactado) pasa a 0, porque 999 días no es una
      distancia real y distorsionaría el escalado. La información no se pierde:
      queda en contactado_antes.
    - contactos_totales: campaign + previous (contactos en esta campaña y antes).
    """
    df = df.copy()
    df["contactado_antes"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, 0)
    df["contactos_totales"] = df["campaign"] + df["previous"]
    return df


def ajustar_imputacion(df_train: pd.DataFrame) -> dict:
    """Calcula con los datos de TRAIN los valores que usaremos para imputar nulos.

    Devuelve un diccionario con:
        moda: moda de cada columna de COLS_MODA.
        educacion_por_job: moda del nivel de estudios para cada tipo de trabajo.
        educacion_global: moda global de education (para cuando job también es nulo).
    """
    return {
        "moda": {c: df_train[c].mode()[0] for c in COLS_MODA},
        "educacion_por_job": df_train.groupby("job")["education"].agg(lambda s: s.mode()[0]).to_dict(),
        "educacion_global": df_train["education"].mode()[0],
    }


def aplicar_imputacion(df: pd.DataFrame, reglas: dict) -> pd.DataFrame:
    """Imputa los nulos de df usando las reglas calculadas con ajustar_imputacion.

    - education: moda de su tipo de trabajo (o la global si job es nulo).
    - job, marital, housing, loan: moda de train.
    - default: los nulos pasan a ser una categoría propia, "desconocido".
    """
    df = df.copy()
    # education se imputa primero, mientras job conserva sus nulos originales
    edu = df["job"].map(reglas["educacion_por_job"]).fillna(reglas["educacion_global"])
    df["education"] = df["education"].fillna(edu)
    for col, valor in reglas["moda"].items():
        df[col] = df[col].fillna(valor)
    df["default"] = df["default"].fillna("desconocido")
    return df


def cargar_datos_procesados():
    """Carga X_train, X_test, y_train e y_test de data/processed.

    Returns:
        Tupla (X_train, X_test, y_train, y_test). Las y son Series (1 = contrata).
    """
    X_train = pd.read_csv(PROCESSED / "X_train.csv")
    X_test = pd.read_csv(PROCESSED / "X_test.csv")
    y_train = pd.read_csv(PROCESSED / "y_train.csv")["y"]
    y_test = pd.read_csv(PROCESSED / "y_test.csv")["y"]
    return X_train, X_test, y_train, y_test


def crear_preprocesador() -> ColumnTransformer:
    """Preprocesado común a todos los modelos.

    - Numéricas: StandardScaler.
    - Categóricas: OneHotEncoder (las categorías nuevas se ignoran en vez de fallar).
    """
    return ColumnTransformer([
        ("num", StandardScaler(), COLS_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COLS_CATEGORICAS),
    ])


def crear_pipeline(modelo) -> Pipeline:
    """Une el preprocesado común con un modelo de scikit-learn.

    Al estar el preprocesado dentro del pipeline, en validación cruzada se ajusta
    solo con los datos de entrenamiento de cada pliegue (sin fugas), y el modelo
    guardado sabe predecir un caso nuevo sin preprocesar a mano.
    """
    return Pipeline([("preprocesado", crear_preprocesador()), ("modelo", modelo)])


def evaluar_cv(pipeline: Pipeline, X, y, n_pliegues: int = 5, semilla: int = 42) -> dict:
    """Valida un pipeline con validación cruzada estratificada y devuelve sus métricas.

    Calcula recall, precisión y F1 en train y en validación (media de los pliegues)
    y el tiempo medio de entrenamiento. Solo se usan datos de TRAIN: el test no se toca.
    """
    cv = StratifiedKFold(n_splits=n_pliegues, shuffle=True, random_state=semilla)
    res = cross_validate(
        pipeline, X, y, cv=cv, n_jobs=-1, return_train_score=True,
        scoring={"recall": "recall", "precision": "precision", "f1": "f1"},
    )
    metricas = {}
    for m in ["recall", "precision", "f1"]:
        metricas[f"{m}_train"] = float(res[f"train_{m}"].mean())
        metricas[f"{m}_val"] = float(res[f"test_{m}"].mean())
    metricas["tiempo_entrenamiento_s"] = float(res["fit_time"].mean())
    return metricas


def calcular_metricas(y_real, y_pred) -> dict:
    """Recall, precisión y F1 de la clase 1 (cliente que contrata)."""
    return {
        "recall": recall_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "f1": f1_score(y_real, y_pred),
    }


def registrar_en_mlflow(nombre: str, params: dict, metricas: dict) -> None:
    """Registra un experimento en MLflow: nombre, hiperparámetros y métricas.

    Args:
        nombre: nombre de la ejecución (p. ej. "regresion_logistica_base").
        params: hiperparámetros y opciones usadas (p. ej. {"C": 1.0, "smote": False}).
        metricas: diccionario devuelto por evaluar_cv.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name=nombre):
        mlflow.log_params(params)
        mlflow.log_metrics(metricas)
