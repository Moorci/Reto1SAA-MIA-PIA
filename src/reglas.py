"""Reglas de negocio (MIA): convierten la probabilidad p en una decisión.

Cómo funciona:
    - Cada regla tiene prioridad, condición, decisión y porqué.
    - Se revisan en orden de prioridad (1 = la primera) y la PRIMERA que se cumple
      decide. Así cada cliente recibe una sola decisión.
    - R1 y R2 son reglas de seguridad: si no podemos fiarnos de p o de los datos,
      no decide la máquina, decide una persona.
    - La última regla (R9) es la regla por defecto: siempre se cumple, para que
      ningún cliente se quede sin decisión.

Las decisiones coinciden con las ramas del flujo de n8n (n8n/flujo.json):
LLAMAR_AHORA, LLAMAR_ESTA_SEMANA, ENVIAR_EMAIL, NO_LLAMAR y REVISION_MANUAL.
"""
import math

# ---------------------------------------------------------------------------
# Umbrales de probabilidad
# ---------------------------------------------------------------------------
# AJUSTAR con las probabilidades del modelo elegido en VALIDACIÓN (no en test).
# Si el modelo se entrenó con SMOTE, sus p salen más altas de lo real y conviene
# subir los umbrales.
TASA_BASE = 0.113   # 11,3 %: tasa de contratación llamando sin criterio (EDA)
UMBRAL_ALTO = 0.50  # a partir de aquí es más probable que contrate que no
UMBRAL_MEDIO = 0.25 # más del doble de la tasa base

# Otros límites
MAX_CONTACTOS_CAMPANA = 10  # a partir de aquí el cliente está saturado
EDAD_MIN, EDAD_MAX = 18, 100
TRABAJOS_PROPENSOS = {"student", "retired"}  # contratan 2-3 veces más (EDA)
CAMPOS_OBLIGATORIOS = ["age", "job", "campaign", "poutcome"]


def p_valida(p) -> bool:
    """Indica si p es un número entre 0 y 1 (y no es NaN)."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return False
    return not math.isnan(p) and 0.0 <= p <= 1.0


def datos_incorrectos(cliente: dict) -> bool:
    """Indica si al cliente le faltan campos obligatorios o tiene valores imposibles."""
    if any(cliente.get(campo) is None for campo in CAMPOS_OBLIGATORIOS):
        return True
    return not (EDAD_MIN <= cliente["age"] <= EDAD_MAX) or cliente["campaign"] < 0


# ---------------------------------------------------------------------------
# Reglas, ordenadas por prioridad
# ---------------------------------------------------------------------------
# condicion(p, cliente) -> bool. Al llegar a R3, R1 ya garantiza que p es válida
# y R2 que los campos obligatorios existen.
REGLAS = [
    {
        "prioridad": 1,
        "regla": "R1",
        "nombre": "Seguridad: probabilidad no válida",
        "condicion": lambda p, c: not p_valida(p),
        "decision": "REVISION_MANUAL",
        "porque": "El modelo no ha dado una probabilidad válida (vacía o fuera de "
                  "0-1). Sin p fiable no se decide automáticamente: lo revisa el supervisor.",
    },
    {
        "prioridad": 2,
        "regla": "R2",
        "nombre": "Seguridad: datos incompletos o imposibles",
        "condicion": lambda p, c: datos_incorrectos(c),
        "decision": "REVISION_MANUAL",
        "porque": f"Faltan datos obligatorios o hay valores imposibles (edad fuera de "
                  f"{EDAD_MIN}-{EDAD_MAX}, contactos negativos). La predicción no es fiable.",
    },
    {
        "prioridad": 3,
        "regla": "R3",
        "nombre": "Cliente saturado de llamadas",
        "condicion": lambda p, c: c["campaign"] >= MAX_CONTACTOS_CAMPANA,
        "decision": "NO_LLAMAR",
        "porque": f"Ya se le ha llamado {MAX_CONTACTOS_CAMPANA} veces o más en esta "
                  "campaña. Insistir apenas suma ventas y desgasta la relación con el cliente, "
                  "aunque el modelo dé una p alta.",
    },
    {
        "prioridad": 4,
        "regla": "R4",
        "nombre": "Probabilidad alta",
        "condicion": lambda p, c: p >= UMBRAL_ALTO,
        "decision": "LLAMAR_AHORA",
        "porque": f"p >= {UMBRAL_ALTO:.0%}: es más probable que contrate que no. Son las "
                  "llamadas más rentables y hay que hacerlas antes de que pierda el interés.",
    },
    {
        "prioridad": 5,
        "regla": "R5",
        "nombre": "Éxito en la campaña anterior",
        "condicion": lambda p, c: c["poutcome"] == "success" and p >= UMBRAL_MEDIO,
        "decision": "LLAMAR_AHORA",
        "porque": "Ya contrató en la campaña anterior: en el EDA esos clientes contratan "
                  f"≈65 % de las veces. Con p >= {UMBRAL_MEDIO:.0%} se le adelanta la llamada.",
    },
    {
        "prioridad": 6,
        "regla": "R6",
        "nombre": "Probabilidad media",
        "condicion": lambda p, c: p >= UMBRAL_MEDIO,
        "decision": "LLAMAR_ESTA_SEMANA",
        "porque": f"p >= {UMBRAL_MEDIO:.0%}, más del doble de la tasa base ({TASA_BASE:.1%}). "
                  "Merece una llamada, pero después de los clientes de prioridad alta.",
    },
    {
        "prioridad": 7,
        "regla": "R7",
        "nombre": "Perfil propenso (estudiante o jubilado)",
        "condicion": lambda p, c: c["job"] in TRABAJOS_PROPENSOS and p >= TASA_BASE,
        "decision": "LLAMAR_ESTA_SEMANA",
        "porque": "Estudiantes (≈32 %) y jubilados (≈26 %) contratan entre 2 y 3 veces más "
                  "que la media (EDA). Si además p supera la tasa base, se le llama.",
    },
    {
        "prioridad": 8,
        "regla": "R8",
        "nombre": "Probabilidad baja pero por encima de la base",
        "condicion": lambda p, c: p >= TASA_BASE,
        "decision": "ENVIAR_EMAIL",
        "porque": f"p entre {TASA_BASE:.1%} y {UMBRAL_MEDIO:.0%}: algo mejor que llamar al "
                  "azar, pero no compensa el coste de una llamada. Un correo es casi gratis.",
    },
    {
        "prioridad": 9,
        "regla": "R9",
        "nombre": "Por defecto: probabilidad muy baja",
        "condicion": lambda p, c: True,
        "decision": "NO_LLAMAR",
        "porque": f"p < {TASA_BASE:.1%}: peor que llamar sin criterio. La llamada casi "
                  "seguro se pierde; ese tiempo se dedica a clientes con más opciones.",
    },
]


def decidir(p, cliente: dict) -> dict:
    """Aplica las reglas por orden de prioridad y devuelve la primera que se cumple.

    Args:
        p: probabilidad de contratar que da el modelo (entre 0 y 1).
        cliente: datos del cliente (al menos los de CAMPOS_OBLIGATORIOS).

    Returns:
        Diccionario con prioridad, regla, nombre, decision, motivo y p. Se pasa
        a src/n8n.py así:
            enviar_decision(cliente_id, r["p"], r["decision"], regla=r["regla"],
                            motivo=r["motivo"], prioridad=r["prioridad"],
                            nombre_regla=r["nombre"])
    """
    for regla in sorted(REGLAS, key=lambda r: r["prioridad"]):
        try:
            se_cumple = regla["condicion"](p, cliente)
        except (KeyError, TypeError, ValueError):
            se_cumple = False
        if se_cumple:
            return {
                "prioridad": regla["prioridad"],
                "regla": regla["regla"],
                "nombre": regla["nombre"],
                "decision": regla["decision"],
                "motivo": regla["porque"],
                "p": float(p) if p_valida(p) else None,
            }
    # No debería llegar aquí (R9 siempre se cumple), pero por seguridad:
    return {"prioridad": 0, "regla": "R0", "nombre": "Sin regla", "decision": "REVISION_MANUAL",
            "motivo": "Ninguna regla se ha cumplido.", "p": None}


def tabla_reglas() -> list[dict]:
    """Devuelve las reglas sin la condición, para mostrarlas en Streamlit o el README."""
    return [
        {"Prioridad": r["prioridad"], "Regla": r["regla"], "Nombre": r["nombre"],
         "Decisión": r["decision"], "Por qué": r["porque"]}
        for r in sorted(REGLAS, key=lambda r: r["prioridad"])
    ]
