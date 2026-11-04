"""Funciones compartidas del proyecto (carga de datos, nulos, variables nuevas...)."""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"
MODELS = RAIZ / "models"
RESULTADOS = RAIZ / "resultados"

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
