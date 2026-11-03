"""Funciones compartidas del proyecto (carga de datos, métricas, etc.)."""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"
MODELS = RAIZ / "models"


def cargar_datos_raw(nombre_archivo: str, sep: str = ";") -> pd.DataFrame:
    """Carga un CSV de data/raw y lo devuelve como DataFrame.

    Args:
        nombre_archivo: nombre del fichero dentro de data/raw.
        sep: separador del CSV (el de Bank Marketing usa ";").
    """
    return pd.read_csv(RAW / nombre_archivo, sep=sep)
