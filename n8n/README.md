# Flujo de n8n

`flujo.json`: **Webhook → Normalizar datos → Switch (una rama por decisión) → acción → respuesta a Streamlit**.

| Decisión (`src/reglas.py`) | Acción en n8n |
|---|---|
| `LLAMAR_AHORA` | Crea un evento en **Google Calendar** dentro de 1 hora para llamar al cliente |
| `LLAMAR_ESTA_SEMANA` | Añade una fila a la hoja `cola_llamadas` de **Google Sheets** |
| `ENVIAR_EMAIL` | Envía al cliente un correo informativo con **Gmail** (en vez de gastar una llamada) |
| `NO_LLAMAR` | Añade una fila a la hoja `no_llamar` de **Google Sheets** |
| `REVISION_MANUAL` | **Regla de seguridad:** envía un correo al supervisor para que decida a mano |
| cualquier otra | Responde con error 400 |

## Puesta en marcha

1. En n8n: *Workflows → Import from file* → `n8n/flujo.json`.
2. Crea las credenciales de Google (Calendar, Sheets y Gmail) y asígnalas a cada nodo.
3. Crea una Google Sheet con dos pestañas, `cola_llamadas` y `no_llamar`, con estas cabeceras en la fila 1:
   `fecha | cliente_id | p | decision | regla | motivo | prioridad | email_cliente`.
   Pega su ID (el trozo de la URL entre `/d/` y `/edit`) en los dos nodos de Sheets.
4. Cambia los correos de prueba de los nodos de Gmail.
5. Activa el flujo. La URL es `http://localhost:5678/webhook/deposito-decision`
   (en pruebas desde el editor: `/webhook-test/deposito-decision`). Si es otra, ponla en la variable de entorno `N8N_WEBHOOK_URL`.

## Probarlo sin Streamlit

```bash
curl -X POST http://localhost:5678/webhook-test/deposito-decision \
  -H "Content-Type: application/json" \
  -d '{"cliente_id":"C-001","p":0.72,"decision":"LLAMAR_AHORA","regla":"R2","motivo":"p alta","prioridad":2}'
```

Desde Python:

```python
from src.n8n import enviar_decision
enviar_decision("C-001", 0.72, "LLAMAR_AHORA", regla="R2", motivo="p alta", prioridad=2)
```
