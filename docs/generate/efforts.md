# 📄 efforts.py

**Ruta relativa:** `generate/efforts.py`
**Ruta absoluta:** `/Users/jumo04/Documents/DFKI/Projects/NFDI4Culture/Developments/own_broker/generate/efforts.py`

## 📖 Descripción del módulo

Niveles de 'esfuerzo de razonamiento' para modelos open-weights con modo thinking
(p. ej. familia Qwen3) mediante *budget forcing* en dos pasadas:
 1) generar el razonamiento con tope de `think_budget` tokens;
 2) si no terminó, cerrar el bloque de razonamiento y pedir la respuesta final.
Es una aproximación operativa a 'esfuerzo' (los modelos propietarios lo exponen nativamente).

## 🔗 Dependencias (imports)

- `from __future__ import annotations`
- `from backends import GenRequest, GenResult`
- `from dataclasses import dataclass, replace`

## 📦 Clases (1)

### 🧊 `EffortSpec`

---

## ⚙️ Funciones globales (1)

### 📝 `run_with_effort(backend, reqs, spec)`

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `backend` | `Any` | — |
| `reqs` | `Any` | — |
| `spec` | `Any` | — |

**Retorna:** `Any`
