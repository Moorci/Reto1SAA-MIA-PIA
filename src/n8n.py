"""Envío de la decisión al webhook de n8n (MIA).

El flujo de n8n (n8n/flujo.json) recibe un POST con este JSON:

    {
        "cliente_id": "C-00123",          # identificador del cliente
        "p": 0.72,                        # probabilidad de contratar (0-1)
        "decision": "LLAMAR_AHORA",       # una de DECISIONES_VALIDAS
        "regla": "R2",                    # regla de src/reglas.py que se ha aplicado
        "motivo": "p >= 0.5 ...",         # el porqué de la regla
        "prioridad": 2,                   # prioridad de la regla
        "email_cliente": "x@correo.com"   # opcional, solo para ENVIAR_EMAIL
    }

y hace una acción distinta según "decision".
"""
import os

import requests

# Decisiones que entiende el flujo de n8n (una rama del Switch por cada una)
DECISIONES_VALIDAS = [
    "LLAMAR_AHORA",        # Google Calendar: llamada agendada dentro de 1 hora
    "LLAMAR_ESTA_SEMANA",  # Google Sheets: hoja "cola_llamadas"
    "ENVIAR_EMAIL",        # Gmail: correo informativo al cliente
    "NO_LLAMAR",           # Google Sheets: hoja "no_llamar"
    "REVISION_MANUAL",     # Gmail: aviso al supervisor (regla de seguridad)
]

# URL del webhook. En pruebas desde el editor de n8n se usa /webhook-test/...;
# con el flujo activado, /webhook/...
URL_POR_DEFECTO = "http://localhost:5678/webhook/deposito-decision"


def enviar_decision(
    cliente_id: str,
    p: float,
    decision: str,
    regla: str = "",
    motivo: str = "",
    prioridad: int = 0,
    email_cliente: str = "",
    url: str | None = None,
    timeout: float = 10,
) -> dict:
    """Envía la decisión de las reglas al webhook de n8n y devuelve su respuesta.

    Args:
        cliente_id: identificador del cliente.
        p: probabilidad de contratar que da el modelo (entre 0 y 1).
        decision: decisión de las reglas; debe estar en DECISIONES_VALIDAS.
        regla: código de la regla aplicada (p. ej. "R1").
        motivo: explicación de por qué se ha tomado la decisión.
        prioridad: prioridad de la regla aplicada.
        email_cliente: correo del cliente (solo se usa en ENVIAR_EMAIL).
        url: URL del webhook. Si no se indica, se usa la variable de entorno
            N8N_WEBHOOK_URL o, en su defecto, URL_POR_DEFECTO.
        timeout: segundos máximos de espera.

    Returns:
        Diccionario con la respuesta de n8n, por ejemplo
        {"ok": True, "decision": "LLAMAR_AHORA", "accion": "Llamada agendada ..."}.
        Si n8n no responde, {"ok": False, "error": "<motivo>"}.
    """
    if decision not in DECISIONES_VALIDAS:
        return {"ok": False, "error": f"Decisión no válida: {decision}"}

    url = url or os.getenv("N8N_WEBHOOK_URL", URL_POR_DEFECTO)
    datos = {
        "cliente_id": str(cliente_id),
        "p": round(float(p), 4),
        "decision": decision,
        "regla": regla,
        "motivo": motivo,
        "prioridad": int(prioridad),
        "email_cliente": email_cliente,
    }
    try:
        respuesta = requests.post(url, json=datos, timeout=timeout)
        try:
            return respuesta.json()
        except ValueError:
            return {"ok": respuesta.ok, "status": respuesta.status_code, "texto": respuesta.text}
    except requests.RequestException as e:
        return {"ok": False, "error": f"No se pudo contactar con n8n: {e}"}
