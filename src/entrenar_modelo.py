"""Entrena el modelo elegido con todo train, lo registra en MLflow y lo guarda.

Modelo elegido: Random Forest sin SMOTE con class_weight="balanced"
(notebooks/03_modelos_iker.ipynb). Es el que mejor equilibra lo que pide el
negocio: recall de validación 0,610 (el segundo más alto, casi igual que el SVM
lineal) con más precisión (0,401 frente a 0,352) y un F1 de 0,483, prácticamente
empatado con el mejor (0,490).

Solo usa TRAIN. El test se evalúa una única vez en la app de Streamlit.

Uso, desde la raíz del proyecto:
    python -m src.entrenar_modelo
"""
import json

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from src.utils import (
    EXPERIMENTO,
    MLFLOW_URI,
    MODELS,
    cargar_datos_procesados,
    crear_pipeline,
    evaluar_cv,
)

NOMBRE_MODELO = "Random Forest (class_weight balanced, sin SMOTE)"
RUTA_MODELO = MODELS / "mejor_modelo.joblib"
RUTA_INFO = MODELS / "mejor_modelo_info.json"

# Mejores hiperparámetros encontrados con GridSearchCV en 03_modelos_iker.ipynb
PARAMS = {
    "n_estimators": 250,
    "max_depth": 16,
    "min_samples_split": 2,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
}


def crear_modelo_elegido():
    """Devuelve el pipeline (preprocesado común + Random Forest) sin entrenar."""
    return crear_pipeline(RandomForestClassifier(n_jobs=-1, **PARAMS))


def entrenar_y_guardar() -> dict:
    """Valida el modelo con CV, lo entrena con todo train, lo guarda y lo registra.

    Returns:
        Diccionario con el nombre, los hiperparámetros y las métricas de validación,
        que también se guarda en models/mejor_modelo_info.json para la app.
    """
    X_train, _, y_train, _ = cargar_datos_procesados()  # el test no se toca

    metricas_cv = evaluar_cv(crear_modelo_elegido(), X_train, y_train)
    pipeline = crear_modelo_elegido().fit(X_train, y_train)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(pipeline, RUTA_MODELO)
    info = {"nombre": NOMBRE_MODELO, "params": PARAMS, "metricas_cv": metricas_cv}
    RUTA_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name="modelo_elegido_random_forest"):
        mlflow.log_params({"modelo": "RandomForestClassifier", "smote": False, **PARAMS})
        mlflow.log_metrics(metricas_cv)
        # El modelo lo creamos nosotros, así que confiamos en el tipo Tree de sklearn
        mlflow.sklearn.log_model(
            pipeline, name="modelo", input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    return info


if __name__ == "__main__":
    info = entrenar_y_guardar()
    print(f"Modelo guardado en {RUTA_MODELO}")
    for nombre, valor in info["metricas_cv"].items():
        print(f"  {nombre}: {valor:.3f}")
