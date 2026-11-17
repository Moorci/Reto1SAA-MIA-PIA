"""App de Streamlit: evalúa con test el modelo elegido y predice un caso nuevo.

Tres pestañas:
    1. Evaluación con test: métricas y matriz de confusión del modelo elegido sobre
       clientes que no ha visto nunca, y qué decidirían las reglas con ellos.
    2. Predecir cliente nuevo: formulario -> probabilidad p -> decisión de las
       reglas (src/reglas.py) -> envío opcional al webhook de n8n (src/n8n.py).
    3. Reglas de negocio: la tabla de reglas en orden de prioridad.

Antes de abrirla hay que entrenar y guardar el modelo:
    python -m src.entrenar_modelo
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.n8n import enviar_decision  # noqa: E402
from src.reglas import TASA_BASE, decidir, tabla_reglas  # noqa: E402
from src.utils import (  # noqa: E402
    MODELS,
    RESULTADOS,
    calcular_metricas,
    cargar_datos_procesados,
    crear_variables_nuevas,
)

RUTA_MODELO = MODELS / "mejor_modelo.joblib"
RUTA_INFO = MODELS / "mejor_modelo_info.json"
RUTA_MATRIZ = RESULTADOS / "matriz_confusion.png"

COLORES_DECISION = {
    "LLAMAR_AHORA": "green",
    "LLAMAR_ESTA_SEMANA": "blue",
    "ENVIAR_EMAIL": "orange",
    "NO_LLAMAR": "gray",
    "REVISION_MANUAL": "red",
}

st.set_page_config(page_title="Predicción de depósitos", layout="wide")


# ---------------------------------------------------------------------------
# Carga (en caché: se hace una sola vez por sesión del servidor)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    """Carga el pipeline entrenado (preprocesado + modelo) y su ficha."""
    modelo = joblib.load(RUTA_MODELO)
    info = json.loads(RUTA_INFO.read_text(encoding="utf-8")) if RUTA_INFO.exists() else {}
    return modelo, info


@st.cache_data
def cargar_datos():
    """Carga los datos procesados (train solo se usa para valores por defecto)."""
    return cargar_datos_procesados()


@st.cache_data
def evaluar_en_test(_modelo, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Predice el test una vez y devuelve probabilidades, predicciones y métricas."""
    p = _modelo.predict_proba(X_test)[:, 1]
    y_pred = (p >= 0.5).astype(int)
    return {
        "p": p,
        "y_pred": y_pred,
        "metricas": calcular_metricas(y_test, y_pred),
        "matriz": confusion_matrix(y_test, y_pred),
    }


def dibujar_matriz(matriz):
    """Dibuja la matriz de confusión con etiquetas de negocio."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(matriz, display_labels=["No contrata", "Contrata"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_xlabel("Predicción del modelo")
    ax.set_ylabel("Realidad")
    ax.set_title("Matriz de confusión (test)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.title("Predicción de contratación de depósito a plazo")
st.caption("Bank Marketing · a qué clientes llamar primero")

if not RUTA_MODELO.exists():
    st.error(
        "No hay modelo guardado en `models/mejor_modelo.joblib`. "
        "Entrénalo primero desde la raíz del proyecto con `python -m src.entrenar_modelo`."
    )
    st.stop()

modelo, info = cargar_modelo()
X_train, X_test, y_train, y_test = cargar_datos()

with st.sidebar:
    st.header("Modelo elegido")
    st.write(f"**{info.get('nombre', type(modelo[-1]).__name__)}**")
    if info.get("params"):
        st.json(info["params"], expanded=False)
    if info.get("metricas_cv"):
        cv = info["metricas_cv"]
        st.write("Validación cruzada (5 pliegues, train):")
        st.write(
            f"- Recall: {cv['recall_val']:.3f}\n"
            f"- Precisión: {cv['precision_val']:.3f}\n"
            f"- F1: {cv['f1_val']:.3f}"
        )

tab_test, tab_nuevo, tab_reglas = st.tabs(
    ["📊 Evaluación con test", "🧑 Predecir cliente nuevo", "📋 Reglas de negocio"]
)

# ---------------------------------------------------------------------------
# 1. Evaluación con test
# ---------------------------------------------------------------------------
with tab_test:
    st.subheader("Evaluación final con test")
    st.write(
        f"{len(X_test):,} clientes que el modelo no ha visto nunca "
        f"({y_test.sum():,} contrataron, {y_test.mean():.1%}). Umbral de clasificación: 0,5."
    )

    res = evaluar_en_test(modelo, X_test, y_test)
    m = res["metricas"]
    tn, fp, fn, tp = res["matriz"].ravel()
    cv = info.get("metricas_cv", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Recall", f"{m['recall']:.3f}",
              delta=f"{m['recall'] - cv['recall_val']:+.3f} vs validación" if cv else None,
              delta_color="off")
    c2.metric("Precisión", f"{m['precision']:.3f}",
              delta=f"{m['precision'] - cv['precision_val']:+.3f} vs validación" if cv else None,
              delta_color="off")
    c3.metric("F1", f"{m['f1']:.3f}",
              delta=f"{m['f1'] - cv['f1_val']:+.3f} vs validación" if cv else None,
              delta_color="off")

    col_matriz, col_texto = st.columns([1, 1.3])
    with col_matriz:
        fig = dibujar_matriz(res["matriz"])
        st.pyplot(fig)
        if st.button("Guardar en resultados/matriz_confusion.png"):
            RESULTADOS.mkdir(exist_ok=True)
            fig.savefig(RUTA_MATRIZ, dpi=150, bbox_inches="tight")
            st.success(f"Guardada en {RUTA_MATRIZ.relative_to(RAIZ)}")
    with col_texto:
        st.markdown("**Qué significan los aciertos y errores para el banco**")
        st.markdown(
            f"- **Verdaderos positivos ({tp:,}):** clientes que contratan y el modelo detecta. "
            f"De cada 100 que contratan, detectamos **{m['recall'] * 100:.0f}**.\n"
            f"- **Falsos negativos ({fn:,}):** clientes que contratarían y no marcamos: "
            "**ventas perdidas**.\n"
            f"- **Falsos positivos ({fp:,}):** clientes marcados que no contratan: "
            "**llamadas gastadas**.\n"
            f"- **Verdaderos negativos ({tn:,}):** llamadas que nos ahorramos con razón."
        )
        marcados = tp + fp
        st.markdown(
            f"Llamando solo a los **{marcados:,}** clientes marcados "
            f"({marcados / len(X_test):.0%} del total) conseguimos **{tp:,}** de las "
            f"{tp + fn:,} contrataciones. De cada 100 llamadas, **{m['precision'] * 100:.0f}** "
            f"acaban en venta, frente a **{TASA_BASE * 100:.0f}** llamando sin criterio."
        )

    st.divider()
    st.subheader("Qué decidirían las reglas de negocio con los clientes de test")
    st.caption(
        "Cada cliente pasa por src/reglas.py con su p. La tasa real de contratación de "
        "cada grupo indica si las reglas ordenan bien las llamadas."
    )
    decisiones = [
        decidir(p, cliente)["decision"]
        for p, cliente in zip(res["p"], X_test.to_dict("records"))
    ]
    resumen = (
        pd.DataFrame({"Decisión": decisiones, "contrata": y_test.values})
        .groupby("Decisión")
        .agg(Clientes=("contrata", "size"), Contratan=("contrata", "sum"),
             **{"Tasa real de contratación": ("contrata", "mean")})
        .reindex(list(COLORES_DECISION))
        .dropna()
        .astype({"Clientes": int, "Contratan": int})
    )
    resumen["% de clientes"] = resumen["Clientes"] / resumen["Clientes"].sum()
    st.dataframe(
        resumen.style.format({"Tasa real de contratación": "{:.1%}", "% de clientes": "{:.1%}"}),
        width="stretch",
    )

# ---------------------------------------------------------------------------
# 2. Predecir un cliente nuevo
# ---------------------------------------------------------------------------
def opciones(col: str) -> list:
    """Categorías vistas en train para una columna, ordenadas."""
    return sorted(X_train[col].dropna().unique().tolist())


def indice(col: str, valor) -> int:
    """Posición de un valor en las opciones de la columna (0 si no está)."""
    ops = opciones(col)
    return ops.index(valor) if valor in ops else 0


with tab_nuevo:
    st.subheader("Predecir un cliente nuevo")
    st.write(
        "Rellena los datos del cliente. El modelo calcula la probabilidad **p** de que "
        "contrate y las reglas de negocio la convierten en una decisión."
    )
    medianas = X_train.median(numeric_only=True)

    with st.form("cliente"):
        st.markdown("**Datos personales**")
        a, b, c, d = st.columns(4)
        age = a.number_input("Edad", min_value=0, max_value=120, value=40)
        job = b.selectbox("Trabajo", opciones("job"), index=indice("job", "admin."))
        marital = c.selectbox("Estado civil", opciones("marital"),
                              index=indice("marital", "married"))
        education = d.selectbox("Estudios", opciones("education"),
                                index=indice("education", "university.degree"))

        st.markdown("**Situación financiera**")
        a, b, c = st.columns(3)
        default = a.selectbox("Crédito en impago", opciones("default"),
                              index=indice("default", "no"))
        housing = b.selectbox("Préstamo hipotecario", opciones("housing"))
        loan = c.selectbox("Préstamo personal", opciones("loan"))

        st.markdown("**Contacto en esta campaña**")
        a, b, c, d = st.columns(4)
        contact = a.selectbox("Tipo de teléfono", opciones("contact"))
        month = b.selectbox("Mes", opciones("month"), index=indice("month", "may"))
        day_of_week = c.selectbox("Día de la semana", opciones("day_of_week"))
        campaign = d.number_input("Contactos en esta campaña", min_value=0, value=1)

        st.markdown("**Campañas anteriores**")
        a, b, c, d = st.columns(4)
        contactado = a.checkbox("Contactado en una campaña anterior")
        pdays = b.number_input("Días desde ese contacto", min_value=0, value=6,
                               help="Solo se usa si fue contactado antes.")
        previous = c.number_input("Contactos anteriores", min_value=0, value=0)
        poutcome = d.selectbox("Resultado anterior", opciones("poutcome"),
                               index=indice("poutcome", "nonexistent"))

        st.markdown("**Contexto económico** (por defecto, la mediana de train)")
        a, b, c, d, e = st.columns(5)
        emp_var_rate = a.number_input("Variación del empleo", value=float(medianas["emp.var.rate"]))
        cons_price_idx = b.number_input("IPC", value=float(medianas["cons.price.idx"]),
                                        format="%.3f")
        cons_conf_idx = c.number_input("Confianza del consumidor",
                                       value=float(medianas["cons.conf.idx"]))
        euribor3m = d.number_input("Euríbor 3 meses", value=float(medianas["euribor3m"]),
                                   format="%.3f")
        nr_employed = e.number_input("Nº de empleados",
                                     value=float(medianas["nr.employed"]))

        calcular = st.form_submit_button("Calcular probabilidad y decisión", type="primary")

    if calcular:
        # Mismo formato que el CSV original: pdays = 999 si no fue contactado antes.
        # crear_variables_nuevas añade contactado_antes y contactos_totales.
        cliente = {
            "age": int(age), "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan, "contact": contact,
            "month": month, "day_of_week": day_of_week, "campaign": int(campaign),
            "pdays": int(pdays) if contactado else 999, "previous": int(previous),
            "poutcome": poutcome, "emp.var.rate": emp_var_rate,
            "cons.price.idx": cons_price_idx, "cons.conf.idx": cons_conf_idx,
            "euribor3m": euribor3m, "nr.employed": nr_employed,
        }
        X_nuevo = crear_variables_nuevas(pd.DataFrame([cliente]))[X_train.columns]
        p = float(modelo.predict_proba(X_nuevo)[0, 1])
        st.session_state["resultado"] = {"cliente": cliente, "p": p,
                                         "decision": decidir(p, cliente)}
        st.session_state.pop("respuesta_n8n", None)

    if "resultado" in st.session_state:
        r = st.session_state["resultado"]
        dec = r["decision"]
        color = COLORES_DECISION.get(dec["decision"], "gray")

        st.divider()
        c1, c2 = st.columns([1, 2])
        c1.metric("Probabilidad de contratar (p)", f"{r['p']:.1%}",
                  delta=f"{(r['p'] - TASA_BASE) * 100:+.1f} pts vs tasa base",
                  delta_color="normal")
        c1.progress(min(max(r["p"], 0.0), 1.0))
        with c2:
            st.markdown(f"### Decisión: :{color}[{dec['decision']}]")
            st.markdown(f"**{dec['regla']} · {dec['nombre']}** (prioridad {dec['prioridad']})")
            st.write(dec["motivo"])

        with st.expander("Enviar la decisión a n8n"):
            a, b = st.columns(2)
            cliente_id = a.text_input("ID del cliente", value="C-00001")
            email = b.text_input("Email del cliente (solo para ENVIAR_EMAIL)")
            if st.button("Enviar al webhook"):
                st.session_state["respuesta_n8n"] = enviar_decision(
                    cliente_id, dec["p"], dec["decision"], regla=dec["regla"],
                    motivo=dec["motivo"], prioridad=dec["prioridad"],
                    nombre_regla=dec["nombre"], email_cliente=email,
                )
            if "respuesta_n8n" in st.session_state:
                resp = st.session_state["respuesta_n8n"]
                (st.success if resp.get("ok") else st.error)("Respuesta de n8n:")
                st.json(resp)

# ---------------------------------------------------------------------------
# 3. Reglas de negocio
# ---------------------------------------------------------------------------
with tab_reglas:
    st.subheader("Reglas de negocio")
    st.write(
        "Se revisan en orden de prioridad y **la primera que se cumple decide**. "
        "R1 y R2 son reglas de seguridad: si no podemos fiarnos de p o de los datos, "
        "decide una persona."
    )
    st.dataframe(pd.DataFrame(tabla_reglas()), hide_index=True, width="stretch")
